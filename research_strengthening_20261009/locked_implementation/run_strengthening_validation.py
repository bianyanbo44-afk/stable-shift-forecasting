"""Resumable supplementary experiment; all original scientific records are read-only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys, time, zipfile
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_innovation_20260924'))
import run_functional_study as old
from stable_shift_forecasting.benchmark_data import load_benchmark_data

OUT=Path(__file__).resolve().parent
P=json.loads((OUT/'protocol.json').read_text())
SIG=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
START=time.monotonic()
DATA={}

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_protected():
    lock=json.loads((OUT/'protocol_lock.json').read_text())
    assert lock['protocol_sha256']==SIG
    for rel,digest in lock['protected_files'].items():assert sha(ROOT/rel)==digest,rel
    portable=ROOT/'checkpoint_manifest.json'
    if portable.exists():checkpoint_hashes=json.loads(portable.read_text())
    else:
        with zipfile.ZipFile(ROOT/'submission_eswa_20261008/upload_ready/09_Anonymized_Reproduction.zip') as z:
            checkpoint_hashes=json.loads(z.read('checkpoint_manifest.json'))
    assert len(checkpoint_hashes)==60
    for rel,digest in checkpoint_hashes.items():assert sha(ROOT/rel)==digest,rel

def checkpoint(cell,base,seed):
    ds,h,folder,_,_=cell
    path=ROOT/folder/ds/f'h{h}'/f'seed{seed}'/base
    selected=json.loads((path/'selected.json').read_text())
    return path/f"lr{selected['lr']:g}"/'best.pt'

def stream(cell,base,seed,phase):
    ds,h,folder,datafile,d=cell
    file=ROOT/datafile
    if phase=='independent':file=ROOT/'data/prepared_v4/eia930_through_august_2026.npz'
    if phase=='withheld':file=OUT/'data'/f'{ds}_entities101_116.npz'
    key=(phase,ds)
    if key not in DATA:DATA[key]=load_benchmark_data(file,ds,d)
    data=DATA[key]
    if phase=='withheld':
        first=data.validation_end;stop=len(data.targets)
        origins=np.arange(first,stop-h+1,24)
        cache=OUT/'withheld_predictions'/ds/f'h{h}'/f'seed{seed}'/base/'source.npz'
        if cache.exists():
            with np.load(cache) as z:assert np.array_equal(origins,z['origins']);prediction=z['prediction']
        else:
            cfg=json.loads((ROOT/folder/'protocol.json').read_text())['config']
            if base=='StableShift':
                model=old.StableShiftForecaster(data.features.shape[1],d,h,context_length=336,**cfg['stable_model'])
            else:model=old.build_upstream(base,d,h,336,cfg['upstream_width'])
            path=checkpoint(cell,base,seed)
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state'])
            model.to(old.f.DEVICE).eval()
            chunks=[]
            with torch.no_grad():
                for i in range(0,len(origins),32):
                    x=torch.tensor(np.stack([data.features[o-336:o] for o in origins[i:i+32]]),device=old.f.DEVICE)
                    chunks.append(model(x).cpu().numpy())
            prediction=np.concatenate(chunks)
            del model;torch.cuda.empty_cache()
            cache.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(cache,origins=origins,prediction=prediction)
            write(cache.with_suffix('.json'),{'source_checkpoint':str(path.relative_to(ROOT)),'source_sha256':sha(path),'data_sha256':sha(file),'protocol_signature':SIG,'fixed_positional_mapping':data.manifest['columns']})
    else:
        cache=old.OUT/phase/ds/f'h{h}'/f'seed{seed}'/base/'frozen/predictions.npz'
        if not cache.exists():
            # Portable review archive: regenerate the common source once from
            # the exact archived checkpoint when the optional local cache is absent.
            regenerated=old.prepare(cell,base,seed,phase)
            return regenerated[:6]
        with np.load(cache) as z:origins=z['origins'].copy();prediction=z['prediction'].copy()
    contexts=[]
    for o in origins:
        values=[]
        for k in range(5):
            end=int(o)-24*k;start=end-h
            obs=data.observed[start:end]
            values.append(float(data.targets[start:end][obs].mean()) if obs.any() else 0.)
            assert end<=o
        contexts.append(values)
    targets=np.stack([data.targets[o:o+h] for o in origins])
    masks=np.stack([data.observed[o:o+h] for o in origins])
    return data,origins,torch.tensor(prediction,device=old.f.DEVICE),torch.tensor(targets,device=old.f.DEVICE),torch.tensor(masks,device=old.f.DEVICE),torch.tensor(contexts,device=old.f.DEVICE)

def operator_configs():
    configs=[]
    for rho in P['budgets']:
        for alpha in P['alphas']:configs.append({'rule':'query','alpha':alpha,'trust':rho})
    for alpha in P['alphas']:configs.append({'rule':'none','alpha':alpha,'trust':None})
    for alpha in P['alphas']:configs.append({'rule':'history','alpha':alpha,'trust':P['primary_budget']})
    return configs

@torch.no_grad()
def batch_operators(data,origins,pred,truth,mask,ctx,configs,save):
    K=len(configs);N,H,d=pred.shape
    if K==0:
        save.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(save,origins=origins,timestamps=data.timestamps[origins].to_numpy().astype('datetime64[ns]'),observed_count=mask.sum((1,2)).cpu().numpy(),errors=np.zeros((0,N,3)),geometry=np.zeros((0,0,7)),max_update_label_index=np.full(N,-1))
        write(save.with_suffix('.json'),{'protocol_signature':SIG,'metrics':[],'batch_seconds':0.,'batch_timing_is_not_single_method_runtime':True,'geometry_columns':['before_mse','after_mse','raw_query_rms','query_rms','query_cap','historical_cap','shrink']})
        return []
    X=old.augmented(pred,ctx)
    C=torch.zeros((K,d,H,H+6),device=pred.device)
    alphas=torch.tensor([c['alpha'] for c in configs],device=pred.device)
    rhos=torch.tensor([c['trust'] or 0. for c in configs],device=pred.device)
    qr=torch.tensor([c['rule']=='query' for c in configs],device=pred.device)
    hr=torch.tensor([c['rule']=='history' for c in configs],device=pred.device)
    std=torch.tensor(data.std,device=pred.device)[None,:,None]
    mase=torch.tensor(data.mase_scale,device=pred.device)[None,:,None]
    observed=mask.sum((1,2)).cpu().numpy()
    values=[];geometry=[];max_labels=[];j=0;max_label=-1
    torch.cuda.synchronize();tic=time.monotonic()
    for i,o in enumerate(origins):
        q=X[i]
        while j<i and int(origins[j])+H<=int(o):
            assert int(origins[j])+H-1<int(o)
            if observed[j]>0:
                z=X[j];valid=mask[j].T
                previous=pred[j].T[None]+torch.einsum('kdhi,di->kdh',C,z)
                residual=(truth[j].T[None]-previous)*valid[None]
                norm=1+z.square().sum(-1)
                D=alphas[:,None,None,None]*residual[:,:,:,None]*z[None,:,None,:]/norm[None,:,None,None]
                v=torch.einsum('kdhi,di->kdh',D,q)
                historical=torch.einsum('kdhi,di->kdh',D,z)
                query_rms=v.square().mean((1,2)).sqrt()
                history_rms=historical.square().mean((1,2)).sqrt()
                qcap=rhos*pred[i].square().mean().sqrt().clamp_min(1.)
                hcap=rhos*pred[j].square().mean().sqrt().clamp_min(1.)
                beta=torch.ones(K,device=pred.device)
                beta=torch.where(qr,(qcap/query_rms.clamp_min(1e-12)).clamp(max=1.),beta)
                beta=torch.where(hr,(hcap/history_rms.clamp_min(1e-12)).clamp(max=1.),beta)
                before_query=torch.einsum('kdhi,di->kdh',C,q)
                before_loss=(residual.square().sum((1,2))/valid.sum())
                C+=D*beta[:,None,None,None]
                after_query=torch.einsum('kdhi,di->kdh',C,q)
                actual=(after_query-before_query).square().mean((1,2)).sqrt()
                after_res=(truth[j].T[None]-pred[j].T[None]-torch.einsum('kdhi,di->kdh',C,z))*valid[None]
                after_loss=after_res.square().sum((1,2))/valid.sum()
                assert bool(torch.all(after_loss<=before_loss+2e-6*before_loss.clamp_min(1.)))
                assert bool(torch.all(actual[qr]<=qcap[qr]+2e-6*qcap[qr].clamp_min(1.)))
                geometry.append(torch.stack([before_loss,after_loss,query_rms,actual,qcap,hcap,beta],-1).cpu().numpy())
                max_label=int(origins[j])+H-1
            j+=1
        output=pred[i].T[None]+torch.einsum('kdhi,di->kdh',C,q)
        error=(output-truth[i].T[None])*std
        valid=mask[i].T[None]
        abs_sum=torch.where(valid,error.abs(),0.).double().sum((1,2))
        sq_sum=torch.where(valid,error.square(),0.).double().sum((1,2))
        scaled=torch.where(valid,error.abs()/mase,0.).double().sum((1,2))
        values.append(torch.stack([abs_sum,sq_sum,scaled],-1).cpu().numpy())
        max_labels.append(max_label)
    torch.cuda.synchronize();seconds=time.monotonic()-tic
    array=np.stack(values,axis=1);geo=np.stack(geometry,axis=1) if geometry else np.zeros((K,0,7))
    assert np.isfinite(array).all() and np.isfinite(geo).all()
    count=int(observed.sum());sums=array.sum(1)
    metrics=[{'config':config,'mae':float(sums[k,0]/count),'rmse':float(np.sqrt(sums[k,1]/count)),'mase':float(sums[k,2]/count),'origins':N,'observed_entries':count,'updates':int(geo.shape[1]),'clipped_updates':int((geo[k,:,6]<1-1e-6).sum()),'query_displacement_p99':float(np.quantile(geo[k,:,3],.99)) if geo.shape[1] else 0.,'query_displacement_max':float(geo[k,:,3].max()) if geo.shape[1] else 0.} for k,config in enumerate(configs)]
    save.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(save,origins=origins,timestamps=data.timestamps[origins].to_numpy().astype('datetime64[ns]'),observed_count=observed,errors=array,geometry=geo,max_update_label_index=np.array(max_labels))
    write(save.with_suffix('.json'),{'protocol_signature':SIG,'metrics':metrics,'batch_seconds':seconds,'batch_timing_is_not_single_method_runtime':True,'geometry_columns':['before_mse','after_mse','raw_query_rms','query_rms','query_cap','historical_cap','shrink']})
    del C;return metrics

def disabled(metrics):
    return {'config':{'disabled':True},'mae':metrics['mae'],'rmse':metrics['rmse'],'mase':metrics['mase'],'updates':0}

def choose(records):return min(enumerate(records),key=lambda pair:(pair[1]['mae'],not pair[1]['config'].get('disabled',False),pair[0]))[1]

def select(cell,base,seed):
    ds,h,*_=cell;dest=OUT/'select'/ds/f'h{h}'/f'seed{seed}'/base
    if (dest/'completed.json').exists():return
    data,origins,pred,truth,mask,ctx=stream(cell,base,seed,'select')
    legacy=old.OUT/'select'/ds/f'h{h}'/f'seed{seed}'/base
    frozen=json.loads((legacy/'frozen/metrics.json').read_text())
    configs=operator_configs()
    if not (dest/'operators.npz').exists():batch_operators(data,origins,pred,truth,mask,ctx,configs,dest/'operators.npz')
    records=json.loads((dest/'operators.json').read_text())['metrics']
    # Audit agreement with independently archived legacy trajectories before interpreting additions.
    for kind,rule in [('query','query'),('nlms','none')]:
        for index,alpha in enumerate([.005,.02,.05,.2]):
            ref=json.loads((legacy/f'{kind}_{index}/metrics.json').read_text())
            actual=next(x for x in records if x['config']['rule']==rule and x['config']['alpha']==alpha and (rule=='none' or x['config']['trust']==.05))
            for metric in ['mae','rmse','mase']:assert np.isclose(ref[metric],actual[metric],rtol=2e-5,atol=1e-7),(ds,h,seed,base,kind,alpha,metric,ref[metric],actual[metric])
    cosa=[]
    for rate in P['cosa_rates']:
        for steps in P['cosa_step_budgets']:
            config={'rate':rate,'steps':steps}
            trial=dest/'cosa_trials'/f'lr{rate:g}_steps{steps}.json'
            if not trial.exists():
                # Reuse exact original candidate records; additions run the unchanged controlled module.
                if (rate,steps) in old.f.CONFIGS:
                    index=old.f.CONFIGS.index((rate,steps));result=json.loads((legacy/f'cosa_{index}/metrics.json').read_text())
                    result['measurement_origin']='archived original validation candidate'
                else:
                    result=old.evaluate(data,origins,pred,truth,mask,ctx,seed,'cosa',config)
                    result['measurement_origin']='supplementary enlarged validation candidate'
                write(trial,{'config':config,**result})
            cosa.append(json.loads(trial.read_text()))
    off=disabled(frozen)
    first=choose(cosa+[off])['config']
    additions=[]
    if not first.get('disabled'):
        if first['steps']==20:additions.append({'rate':first['rate'],'steps':50})
        if first['rate']==.005:additions.append({'rate':.01,'steps':first['steps']})
    for config in additions:
        trial=dest/'cosa_trials'/f"lr{config['rate']:g}_steps{config['steps']}.json"
        if not trial.exists():
            result=old.evaluate(data,origins,pred,truth,mask,ctx,seed,'cosa',config)
            write(trial,{'config':config,**result,'measurement_origin':'predeclared validation-only boundary extension'})
        cosa.append(json.loads(trial.read_text()))
    choices={'query_expanded':choose([x for x in records if x['config']['rule']=='query' and x['config']['trust']==.05]+[off]),
             'nlms_expanded':choose([x for x in records if x['config']['rule']=='none']+[off]),
             'history_clip':choose([x for x in records if x['config']['rule']=='history']+[off]),
             'cosa_expanded':choose(cosa+[off])}
    choices['budget_by_rho']={str(rho):choose([x for x in records if x['config']['rule']=='query' and x['config']['trust']==rho]+[off]) for rho in P['budgets']}
    write(dest/'selection.json',choices)
    write(dest/'completed.json',{'protocol_signature':SIG,'search_amendment_sha256':sha(OUT/'validation_search_amendment.json'),'checkpoint_sha256':sha(checkpoint(cell,base,seed)),'legacy_operator_metric_replay_passed':True,'selection_outcomes':'original validation only','cosa_active_candidates':len(cosa)})
    print(json.dumps({'stage':'select','dataset':ds,'horizon':h,'source':base,'seed':seed,'selected':{k:v['config'] for k,v in choices.items() if k!='budget_by_rho'}}),flush=True)

def evaluation_configs(selected):
    named={}
    for method in ['query_expanded','nlms_expanded','history_clip']:
        if not selected[method]['config'].get('disabled'):named[method]=selected[method]['config']
    q=selected['query_expanded']['config']
    if not q.get('disabled'):
        named['nlms_matched_new']={'rule':'none','alpha':q['alpha'],'trust':None}
        named['history_matched_new']={'rule':'history','alpha':q['alpha'],'trust':.05}
        for rho in P['budgets']:named[f'budget_fixed_{rho:g}']={'rule':'query','alpha':q['alpha'],'trust':rho}
    for rho in P['budgets']:
        c=selected['budget_by_rho'][str(rho)]['config']
        if not c.get('disabled'):named[f'budget_tuned_{rho:g}']=c
    # Deduplicate identical recurrences while retaining all named reports.
    configs=[];mapping={}
    for method,config in named.items():
        if config not in configs:configs.append(config)
        mapping[method]=configs.index(config)
    for method in ['query_expanded','nlms_expanded','history_clip','nlms_matched_new','history_matched_new']:
        mapping.setdefault(method,-1)
    for rho in P['budgets']:
        mapping.setdefault(f'budget_fixed_{rho:g}',-1);mapping.setdefault(f'budget_tuned_{rho:g}',-1)
    return configs,mapping

def evaluate(cell,base,seed,phase):
    ds,h,*_=cell;dest=OUT/phase/ds/f'h{h}'/f'seed{seed}'/base
    if (dest/'completed.json').exists():return
    selected=json.loads((OUT/'select'/ds/f'h{h}'/f'seed{seed}'/base/'selection.json').read_text())
    data,origins,pred,truth,mask,ctx=stream(cell,base,seed,phase)
    dest.mkdir(parents=True,exist_ok=True)
    if not (dest/'frozen/metrics.json').exists():old.evaluate(data,origins,pred,truth,mask,ctx,seed,'frozen',{},save=dest/'frozen')
    configs,mapping=evaluation_configs(selected)
    if not (dest/'operators.npz').exists():batch_operators(data,origins,pred,truth,mask,ctx,configs,dest/'operators.npz')
    write(dest/'method_mapping.json',mapping)
    c=selected['cosa_expanded']['config']
    if not (dest/'cosa_expanded/metrics.json').exists():old.evaluate(data,origins,pred,truth,mask,ctx,seed,'cosa',c,save=dest/'cosa_expanded')
    write(dest/'completed.json',{'protocol_signature':SIG,'phase':phase,'source_checkpoint_sha256':sha(checkpoint(cell,base,seed)),'selected_configurations_sha256':sha(OUT/'select'/ds/f'h{h}'/f'seed{seed}'/base/'selection.json'),'data_manifest':data.manifest,'scoring_policy':'all observed target entries','source_retraining':False})
    metrics=json.loads((dest/'operators.json').read_text())['metrics'];fr=json.loads((dest/'frozen/metrics.json').read_text());co=json.loads((dest/'cosa_expanded/metrics.json').read_text())
    print(json.dumps({'stage':phase,'dataset':ds,'horizon':h,'source':base,'seed':seed,'mae':{'frozen':fr['mae'],'cosa_expanded':co['mae'],**{k:(metrics[i]['mae'] if i>=0 else fr['mae']) for k,i in mapping.items() if k in P['primary_methods']}}}),flush=True)

def lock_selections():
    files=sorted((OUT/'select').glob('*/h*/seed*/*/selection.json'))
    assert len(files)==60,len(files)
    lock={'locked_utc':datetime.now(timezone.utc).isoformat(),'protocol_signature':SIG,'search_amendment_sha256':sha(OUT/'validation_search_amendment.json'),'runner_sha256':sha(Path(__file__)),'files':{p.relative_to(OUT).as_posix():sha(p) for p in files},'withheld_entities_prepared':False}
    p=OUT/'selection_lock.json'
    if p.exists():assert json.loads(p.read_text())['files']==lock['files']
    else:write(p,lock)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['select','test','independent','withheld','lock'],required=True);parser.add_argument('--dataset');parser.add_argument('--seed',type=int);parser.add_argument('--horizon',type=int)
    args=parser.parse_args();verify_protected()
    if args.stage=='lock':lock_selections();return
    if args.stage!='select':
        lock=json.loads((OUT/'selection_lock.json').read_text())
        for rel,digest in lock['files'].items():assert sha(OUT/rel)==digest,rel
    if args.stage=='withheld':
        assert (OUT/'withheld_data_manifest.json').exists()
        assert json.loads((OUT/'withheld_data_manifest.json').read_text())['selection_lock_sha256']==sha(OUT/'selection_lock.json')
    try:
        for cell in old.f.CELLS:
            ds,h,*_=cell
            if args.dataset and ds!=args.dataset:continue
            if args.horizon and h!=args.horizon:continue
            if args.stage=='independent' and ds!='eia930':continue
            if args.stage=='withheld' and ds=='eia930':continue
            for seed in ([args.seed] if args.seed else P['seeds']):
                for base in P['sources']:
                    if time.monotonic()-START>780:raise TimeoutError('Resumable process chunk complete')
                    if args.stage=='select':select(cell,base,seed)
                    else:evaluate(cell,base,seed,args.stage)
    except TimeoutError as e:
        verify_protected();print(str(e),flush=True);sys.exit(75)
    verify_protected();print('STAGE COMPLETE',args.stage,round(time.monotonic()-START,1),flush=True)

if __name__=='__main__':main()
