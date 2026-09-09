"""Train one shared forecaster on five regions and test two unseen regions.

PJM, MISO, CISO, NYIS and ISNE supply training and validation windows. ERCO and
SWPP never enter training or model selection. Their scale is initialized from
the 336 observed hours immediately before the first 2026 forecast and then
frozen. Every method uses the same information and per-entity online state.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset,DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import run_benchmark_v3 as bench
from stable_shift_forecasting.benchmark_data import BenchmarkData
from stable_shift_forecasting.data import _calendar_features
from stable_shift_forecasting.model import StableShiftForecaster

SEEN=['PJM','MISO','CISO','NYIS','ISNE'];UNSEEN=['ERCO','SWPP']


def entity_data(path,region,length=336):
    with np.load(path,allow_pickle=False) as z:
        names=z['columns'].astype(str).tolist();col=names.index(region)
        stamps=pd.DatetimeIndex(z['timestamps']);raw=z['values'][:,col:col+1].astype(np.float32)
        observed=z['observed'][:,col:col+1].astype(bool)&np.isfinite(raw)
    raw=np.where(observed,raw,np.nan)
    train_end=int(stamps.searchsorted(pd.Timestamp('2025-01-01 01:00')))
    val_end=int(stamps.searchsorted(pd.Timestamp('2026-01-01 01:00')))
    calibration=raw[:train_end] if region in SEEN else raw[val_end-length:val_end]
    mean=np.nanmean(calibration,axis=0);std=np.nanstd(calibration,axis=0)
    std=np.where(std>1e-6,std,1.).astype(np.float32)
    filled=pd.DataFrame(raw).ffill().to_numpy(np.float32)
    warmup=int(np.flatnonzero(np.isfinite(filled).all(1))[0]);filled[:warmup]=np.nan_to_num(filled[:warmup])
    target=(filled-mean)/std;calendar=_calendar_features(stamps)
    pairs=observed[24:train_end]&observed[:train_end-24]
    scale=np.nanmean(np.where(pairs,np.abs(raw[24:train_end]-raw[:train_end-24]),np.nan),axis=0)
    manifest={'dataset':f'eia930_{region}','source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'region':region,'seen_during_training':region in SEEN,
              'normalization':'observed training data' if region in SEEN else '336 hours immediately before first test forecast, then frozen',
              'train_end_exclusive':str(stamps[train_end]),'validation_end_exclusive':str(stamps[val_end]),
              'first_timestamp':str(stamps[0]),'last_timestamp':str(stamps[-1]),
              'mean':mean.tolist(),'std':std.tolist()}
    return BenchmarkData(np.c_[target,calendar].astype(np.float32),target.astype(np.float32),calendar,
                         observed,stamps,mean,std,scale,train_end,val_end,warmup,manifest)


def run(config,methods,horizons,seeds):
    torch.set_num_threads(config['num_threads']);device=config['device']
    path=ROOT/config['source_data'];out=ROOT/config['output'];out.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256((bench.source_digest()+Path(__file__).read_text()+json.dumps(config,sort_keys=True)).encode()).hexdigest()
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())['signature']!=signature:
        raise RuntimeError('Transfer source/config changed; choose a fresh output')
    bench.write_json(out/'protocol.json',{'signature':signature,'config':config,'torch':torch.__version__,
                                       'seen':SEEN,'unseen':UNSEEN,'selection_metric':'Pooled standardized validation MAE over seen regions'})
    snapshot=out/'source_snapshot/scripts';snapshot.mkdir(parents=True,exist_ok=True)
    for source in [Path(__file__),ROOT/'scripts/run_benchmark_v3.py']:shutil.copy2(source,snapshot/source.name)
    seen=[entity_data(path,r) for r in SEEN];heldout=[entity_data(path,r) for r in UNSEEN]
    proxy=SimpleNamespace(features=np.zeros((1,7)),targets=np.zeros((1,1)),mean=np.zeros(1),std=np.ones(1),
                          manifest={'dataset':'eia930_seen_regions','regions':SEEN,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for horizon in horizons:
        L=config['context_length']
        train=ConcatDataset([d.windows('train',L,horizon,config['train_stride']) for d in seen])
        validation=ConcatDataset([d.windows('validation',L,horizon,config['validation_stride']) for d in seen])
        reference=bench.drift_reference(proxy,train,validation,config)
        for d in heldout:
            folder=out/d.manifest['dataset']/f'h{horizon}'
            bench.write_json(out/d.manifest['dataset']/'data_manifest.json',d.manifest)
            test=d.windows('test',L,horizon,horizon,include_unobserved=True)
            bench.write_json(folder/'origins.json',{'test':test.origins.tolist()})
            bench.write_json(folder/'thresholds.json',{'drift_threshold':reference[2],'shock_threshold':reference[3]})
        for seed in seeds:
            for method in methods:
                model=None;initial=None;shared=out/'shared'/f'h{horizon}'/f'seed{seed}'/method
                if method in ['SeasonalNaive','WeeklyNaive']:
                    period=168 if method=='WeeklyNaive' else 24;total=0.;count=0
                    for batch in DataLoader(validation,batch_size=config['batch_size']):
                        x=batch['context'][:,-period:,:1]
                        y=x.repeat(1,(horizon+period-1)//period,1)[:,:horizon]
                        error=(y-batch['target']).abs();total+=float(error.sum());count+=error.numel()
                    selection={'method':method,'best_validation_mae':total/count,'parameters':0,'signature':signature}
                else:
                    trials=[bench.train_trial(method,proxy,train,validation,horizon,seed,lr,config,shared/f'lr{lr:g}',signature)
                            for lr in config['learning_rates']]
                    selection=min(trials,key=lambda r:r['best_validation_mae'])
                    model=bench.make_model(method,proxy,horizon,config).to(device)
                    model.load_state_dict(torch.load(ROOT/selection['checkpoint'],map_location=device,weights_only=False)['state'])
                    model.eval();initial=copy.deepcopy(model.state_dict())
                for d in heldout:
                    folder=out/d.manifest['dataset']/f'h{horizon}'/f'seed{seed}'/method
                    done=folder/'completed.json'
                    if done.exists():
                        if json.loads(done.read_text())['signature']!=signature:raise RuntimeError('Source mismatch')
                        continue
                    bench.write_json(folder/'selected.json',selection)
                    test=d.windows('test',L,horizon,horizon,include_unobserved=True);summaries=[]
                    variants=['online','no_update','no_filter','anchor_only'] if method=='StableShift' else (['online'] if method.startswith('StableShift') else ['frozen'])
                    for variant in variants:
                        for scenario in config['scenarios']:
                            if initial is not None:
                                model.load_state_dict(initial)
                                if isinstance(model,StableShiftForecaster):
                                    model.robust_filter=config['stable_model']['robust_filter'];model.robust_clip=config['stable_model']['robust_clip']
                                    if variant=='no_filter':model.robust_filter=False;model.robust_clip=0.
                            frame,predictions,peak=bench.evaluate_stream(method,model,test,d,scenario,config,reference,
                                                                         'online' if variant=='no_filter' else variant)
                            for key,value in {'dataset':d.manifest['dataset'],'horizon':horizon,'seed':seed,
                                               'method':method,'variant':variant,'scenario':scenario}.items():frame[key]=value
                            dest=folder/variant/scenario;dest.mkdir(parents=True,exist_ok=True)
                            frame.to_csv(dest/'windows.csv',index=False)
                            np.savez_compressed(dest/'predictions.npz',prediction=predictions,origins=test.origins)
                            summaries.append({'variant':variant,'scenario':scenario,'mae':float(frame.mae.mean()),
                                              'rmse':float(np.sqrt(frame.mse.mean())),'peak_gpu_mb':peak})
                            print(json.dumps({'region':d.manifest['region'],'method':method,'seed':seed,'horizon':horizon,
                                              'variant':variant,'scenario':scenario,'mae':float(frame.mae.mean())}),flush=True)
                    bench.write_json(done,{'signature':signature,'selection':selection,'summaries':summaries})
                del model,initial
                if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/entity_transfer_v3.json')
    p.add_argument('--methods',nargs='+');p.add_argument('--horizons',nargs='+',type=int);p.add_argument('--seeds',nargs='+',type=int)
    a=p.parse_args();config=json.loads((ROOT/a.config).read_text())
    run(config,a.methods or config['methods'],a.horizons or config['horizons'],a.seeds or config['seeds'])
