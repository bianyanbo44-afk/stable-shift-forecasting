"""Locked functional-calibration study; resumable source/checkpoint reuse.

The proposed rule is a query-constrained normalized-LMS update in the effective
output map. Classical NLMS is the matched unconstrained control. COSA uses its
reviewed output module under the same causal, daily matured-feedback schedule.
"""
from pathlib import Path
import argparse,copy,hashlib,json,sys,time
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pilot_cosa_factorial as f
from stable_shift_forecasting.benchmark_data import load_benchmark_data
from stable_shift_forecasting.model import StableShiftForecaster
from stable_shift_forecasting.upstream_baselines import build_upstream

ROOT=f.ROOT;OUT=ROOT/'results/functional_calibration_20260924';OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[11,22,33,44,55];ALPHAS=[.005,.02,.05,.2];TRUST=.05
START=time.monotonic()
PROTOCOL={
 'date':'2026-09-24','study':'Function-equivalent initialization and query-constrained calibration',
 'source_models':['StableShift','DLinear'],'seeds':SEEDS,'horizons':[24,168],
 'calibration':'Cold-start output adaptation only; no fitted warm start in main comparison',
 'issuance':'Daily origins for both horizons; every issued forecast immutable',
 'feedback':'One previous daily forecast becomes usable when its complete horizon has elapsed; observed entries only; target index strictly earlier than current origin',
 'context':'Five historical H-hour observed-load means ending at t, t-24,...,t-96, available before issuance; identical for all calibration methods',
 'selection':'Second half of the original validation period; observed-entry MAE; independent per seed/source/dataset/horizon; same candidate count for competing methods',
 'cosa_candidates':[{'rate':r,'steps':s} for r,s in f.CONFIGS]+[{'disabled':True}],
 'query_candidates':[{'alpha':a,'trust':TRUST} for a in ALPHAS]+[{'disabled':True}],
 'nlms_candidates':[{'alpha':a,'trust':None} for a in ALPHAS]+[{'disabled':True}],
 'regularization':'COSA uses pinned native module, Adam, dynamic clipping, CALR and L2=1e-4; NLMS denominator adds one',
 'query_budget':'RMS change of current issued normalized forecast <=0.05*max(1,RMS(frozen forecast)) per observed-feedback event',
 'coordinate_probe':'Same gated module and native selected hyperparameters; residual weights scaled by 0.1,1,10 with zero gate, exactly identical initial forecasts',
 'scoring':'Sum observed absolute/squared errors divided by observed forecast-target count; no imputed labels; complete-block sensitivity retained',
 'test_periods':'Original held-out periods; new independent EIA 2026-07-01 01:00 through 2026-09-01 00:00 UTC',
 'independent_holdout_status_at_lock':'Not yet downloaded or inspected',
 'statistics':'Paired seed/circular-time-block bootstrap of error-sum/count ratios; 5000 draws; 7-day primary and 14/28-day sensitivity',
 'main_multiplicity':'Holm over QNC versus frozen/COSA/NLMS x 12 benchmark cells x MAE/RMSE =72 tests, separately per block duration; independent EIA family separate',
 'provenance':'COSA source commit 527c0feb9e997dd85af485ee027616b446e4ae77; exact CC BY-NC-SA 4.0 license retained; no claim to reproduce native retrospectively adjusted scores',
 'resource_limit':'Four CPU threads, half GPU memory, at most 900 seconds per process; completed cells resumable',
 'novelty_scope':'Study forecast-equivalent sensitivity and a current-query movement constraint; do not claim NLMS or generic optimizer-coordinate dependence as newly invented'
}
SIGNATURE=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
if (OUT/'protocol.json').exists():
    assert json.loads((OUT/'protocol.json').read_text())['signature']==SIGNATURE
else:f.write(OUT/'protocol.json',{'signature':SIGNATURE,**PROTOCOL})


def budget():
    if time.monotonic()-START>900:raise TimeoutError('Fixed process budget reached; completed cells are saved')


def augmented(p,ctx):
    return torch.cat([p.transpose(1,2),ctx[:,None].expand(-1,p.shape[-1],-1),torch.ones((len(p),p.shape[-1],1),device=p.device)],-1)


class QueryCalibration:
    def __init__(self,h,d,alpha,trust):
        self.C=torch.zeros((d,h,h+6),device=f.DEVICE)
        self.alpha=alpha;self.trust=trust;self.count=0;self.audit=[]

    def forecast(self,p,ctx):
        return p+torch.einsum('dhi,bdi->bdh',self.C,augmented(p,ctx)).transpose(1,2)

    def update(self,p,truth,mask,ctx,current,current_ctx):
        if not bool(mask.any()):return
        x=augmented(p,ctx)[0]
        old=self.forecast(p,ctx)[0]
        residual=((truth[0]-old)*mask[0]).T
        delta=self.alpha*residual[:,:,None]*x[:,None,:]/(1+x.square().sum(-1))[:,None,None]
        query=augmented(current,current_ctx)[0]
        change=torch.einsum('dhi,di->dh',delta,query)
        raw_rms=change.square().mean().sqrt()
        cap=(self.trust*current.square().mean().sqrt().clamp_min(1.)) if self.trust is not None else raw_rms.new_tensor(float('inf'))
        scale=(cap/raw_rms.clamp_min(1e-12)).clamp(max=1.)
        before_prediction=self.forecast(current,current_ctx)
        before_loss=float(((old-truth[0]).square()*mask[0]).sum()/mask[0].sum())
        self.C+=delta*scale
        actual=float((self.forecast(current,current_ctx)-before_prediction).square().mean().sqrt())
        after_loss=float(((self.forecast(p,ctx)[0]-truth[0]).square()*mask[0]).sum()/mask[0].sum())
        assert after_loss<=before_loss+2e-6*max(1.,before_loss)
        if self.trust is not None:assert actual<=float(cap)+2e-6*max(1.,float(cap))
        self.audit.append({'before_mse':before_loss,'after_mse':after_loss,'raw_query_rms':float(raw_rms),
          'query_rms':actual,'cap':float(cap) if self.trust is not None else None,'shrink':float(scale)})
        self.count+=1


def prepare(cell,base,seed,phase):
    ds,h,root,datafile,d=cell
    if phase=='independent':
        datafile='data/prepared_v4/eia930_through_august_2026.npz'
        assert ds=='eia930'
    data=load_benchmark_data(ROOT/datafile,ds,d)
    md=ROOT/root/ds/f'h{h}'/f'seed{seed}'/base
    selected=json.loads((md/'selected.json').read_text())
    checkpoint=md/f"lr{selected['lr']:g}"/'best.pt'
    cfg=json.loads((ROOT/root/'protocol.json').read_text())['config']
    if base=='StableShift':model=StableShiftForecaster(data.features.shape[1],d,h,context_length=336,**cfg['stable_model'])
    else:model=build_upstream(base,d,h,336,cfg['upstream_width'])
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state'])
    model.to(f.DEVICE).eval()
    for p in model.parameters():p.requires_grad_(False)
    if phase=='select':
        first=data.train_end+24*((data.validation_end-data.train_end)//48);stop=data.validation_end
    elif phase=='test':
        first=data.validation_end;stop=len(data.targets)
    else:
        first=int(data.timestamps.searchsorted(pd.Timestamp('2026-07-01 01:00')))
        stop=int(data.timestamps.searchsorted(pd.Timestamp('2026-09-01 01:00')))
        old=load_benchmark_data(ROOT/'data/prepared_v2/eia930_2023_2026.npz',ds,d)
        assert np.array_equal(data.mean,old.mean) and np.array_equal(data.std,old.std)
    origins=np.arange(first,stop-h+1,24)
    predictions=[]
    with torch.no_grad():
        for i in range(0,len(origins),32):
            budget()
            x=torch.tensor(np.stack([data.features[o-336:o] for o in origins[i:i+32]]),device=f.DEVICE)
            predictions.append(model(x))
    pred=torch.cat(predictions).detach()
    targets=torch.tensor(np.stack([data.targets[o:o+h] for o in origins]),device=f.DEVICE)
    mask=torch.tensor(np.stack([data.observed[o:o+h] for o in origins]),device=f.DEVICE)
    contexts=[]
    for o in origins:
        values=[]
        for k in range(5):
            end=int(o)-24*k;start=end-h
            obs=data.observed[start:end]
            values.append(float(data.targets[start:end][obs].mean()) if obs.any() else 0.)
            assert end-1<o
        contexts.append(values)
    ctx=torch.tensor(contexts,device=f.DEVICE)
    del model;torch.cuda.empty_cache()
    return data,origins,pred,targets,mask,ctx,checkpoint


def evaluate(data,origins,pred,truth,mask,ctx,seed,kind,config,scale=1.,save=None):
    h,d=pred.shape[1:];n=len(origins)
    enabled=not config.get('disabled',False) and kind!='frozen'
    if enabled and kind=='cosa':
        torch.manual_seed(seed)
        adapter=f.COSA(h,buffer_context_size=5,n_vars=d,var_wise_gating=True,adapter_type='Linear').to(f.DEVICE)
        with torch.no_grad():
            for layer in adapter.fc_layers:layer.weight.mul_(scale);layer.bias.mul_(scale)
            assert float((f.fast_forward(adapter,pred,ctx)-pred).abs().max())==0.
        updater=f.Updater(adapter,float(config['rate']),int(config['steps']))
        parameters=sum(p.numel() for p in adapter.parameters())
    elif enabled:
        updater=QueryCalibration(h,d,float(config['alpha']),config.get('trust'))
        parameters=updater.C.numel()
    else:updater=None;parameters=0
    std=torch.tensor(data.std,device=f.DEVICE);mase=torch.tensor(data.mase_scale,device=f.DEVICE)
    rows=[];predictions=[];events=[];j=0;max_label=-1
    torch.cuda.synchronize();tic=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for i,o in enumerate(origins):
        budget();o=int(o)
        while j<i and int(origins[j])+h<=o:
            assert int(origins[j])+h-1<o
            if updater is not None:
                if kind=='cosa':updater.update(pred[j:j+1],truth[j:j+1],mask[j:j+1],ctx[j:j+1])
                else:updater.update(pred[j:j+1],truth[j:j+1],mask[j:j+1],ctx[j:j+1],pred[i:i+1],ctx[i:i+1])
                max_label=int(origins[j])+h-1
                events.append({'issue_origin':o,'feedback_origin':int(origins[j]),'max_label_index':max_label,
                               'observed_entries':int(mask[j].sum())})
            j+=1
        with torch.no_grad():
            if updater is None:p=pred[i]
            elif kind=='cosa':p=f.fast_forward(updater.model,pred[i:i+1],ctx[i:i+1])[0]
            else:p=updater.forecast(pred[i:i+1],ctx[i:i+1])[0]
            # Prediction has been issued before this origin's truth is scored.
            e=(p-truth[i])*std;valid=mask[i];count=int(valid.sum())
            absolute=float(torch.where(valid,e.abs(),0.).double().sum())
            squared=float(torch.where(valid,e.square(),0.).double().sum())
            scaled=float(torch.where(valid,e.abs()/mase,0.).double().sum())
            rows.append({'origin':o,'timestamp':str(data.timestamps[o]),'observed_count':count,
               'absolute_error_sum':absolute,'squared_error_sum':squared,'scaled_error_sum':scaled,
               'complete_block':bool(valid.all()),'max_update_label_index':max_label})
            predictions.append(p.cpu().numpy())
    torch.cuda.synchronize();seconds=time.monotonic()-tic
    frame=pd.DataFrame(rows);count=frame.observed_count.sum()
    result={'mae':float(frame.absolute_error_sum.sum()/count),
       'rmse':float(np.sqrt(frame.squared_error_sum.sum()/count)),
       'mase':float(frame.scaled_error_sum.sum()/count),'seconds':seconds,
       'origins':n,'observed_entries':int(count),'complete_blocks':int(frame.complete_block.sum()),
       'post_feedback_observed_entries':int(frame.loc[frame.origin>=origins[0]+h,'observed_count'].sum()),
       'updates':0 if updater is None else updater.count,'mutable_parameters':parameters,
       'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
    if save is not None:
        save.mkdir(parents=True,exist_ok=True)
        frame.to_csv(save/'windows.csv',index=False)
        np.savez_compressed(save/'predictions.npz',origins=origins,prediction=np.stack(predictions))
        pd.DataFrame(events).to_csv(save/'events.csv',index=False)
        if updater is not None and kind!='cosa':pd.DataFrame(updater.audit).to_csv(save/'update_geometry.csv',index=False)
        f.write(save/'metrics.json',result)
    return result


def cell_run(cell,base,seed,phase):
    ds,h,root,datafile,d=cell
    dest=OUT/phase/ds/f'h{h}'/f'seed{seed}'/base
    if (dest/'completed.json').exists():
        assert json.loads((dest/'completed.json').read_text())['signature']==SIGNATURE
        print('cached',phase,ds,h,seed,base,flush=True);return
    data,origins,pred,truth,mask,ctx,checkpoint=prepare(cell,base,seed,phase)
    digest=f.sha(checkpoint);dest.mkdir(parents=True,exist_ok=True)
    if phase=='select':
        no=evaluate(data,origins,pred,truth,mask,ctx,seed,'frozen',{},save=dest/'frozen')
        selections={};trials=[]
        for kind,candidates in [('cosa',PROTOCOL['cosa_candidates']),('query',PROTOCOL['query_candidates']),('nlms',PROTOCOL['nlms_candidates'])]:
            records=[]
            for index,config in enumerate(candidates):
                if config.get('disabled'):result=no
                else:result=evaluate(data,origins,pred,truth,mask,ctx,seed,kind,config,save=dest/f'{kind}_{index}')
                records.append({'index':index,'config':config,**result})
                trials.append({'kind':kind,'index':index,'config':json.dumps(config),**result})
            selected=min(records,key=lambda x:(x['mae'],not x['config'].get('disabled',False),x['index']))
            selections[kind]=selected
        f.write(dest/'selection.json',selections);pd.DataFrame(trials).to_csv(dest/'trials.csv',index=False)
        print(json.dumps({'phase':phase,'dataset':ds,'h':h,'seed':seed,'base':base,
            'chosen':{k:{'mae':v['mae'],'config':v['config']} for k,v in selections.items()}}),flush=True)
    else:
        choice=OUT/'select'/ds/f'h{h}'/f'seed{seed}'/base/'selection.json'
        selected=json.loads(choice.read_text());summaries={}
        tasks=[('frozen','frozen',{},1.),('cosa','cosa',selected['cosa']['config'],1.),
               ('cosa_scale_01','cosa',selected['cosa']['config'],.1),
               ('cosa_scale_10','cosa',selected['cosa']['config'],10.),
               ('query','query',selected['query']['config'],1.),('nlms','nlms',selected['nlms']['config'],1.)]
        for name,kind,config,scale in tasks:
            summaries[name]=evaluate(data,origins,pred,truth,mask,ctx,seed,kind,config,scale,save=dest/name)
        f.write(dest/'summary.json',summaries)
        print(json.dumps({'phase':phase,'dataset':ds,'h':h,'seed':seed,'base':base,
                         'mae':{k:v['mae'] for k,v in summaries.items()}}),flush=True)
    assert f.sha(checkpoint)==digest
    f.write(dest/'completed.json',{'signature':SIGNATURE,'runner_sha256':f.sha(Path(__file__)),
      'source_checkpoint':str(checkpoint.relative_to(ROOT)),'source_sha256':digest,
      'data_manifest':data.manifest,'first_issue':str(data.timestamps[origins[0]]),
      'last_issue':str(data.timestamps[origins[-1]]),'all_update_labels_precede_origin':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['select','test','independent'],required=True)
    p.add_argument('--seed',type=int);p.add_argument('--dataset');args=p.parse_args()
    for cell in f.CELLS:
        if args.dataset and cell[0]!=args.dataset:continue
        if args.phase=='independent' and cell[0]!='eia930':continue
        for seed in ([args.seed] if args.seed else SEEDS):
            for base in f.BASES:cell_run(cell,base,seed,args.phase)
    print('PHASE COMPLETED',args.phase,time.monotonic()-START,flush=True)
