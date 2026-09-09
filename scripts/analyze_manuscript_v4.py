"""Rebuild manuscript evidence from saved v3 streams, without running training.

All completed streams are inventoried. The balanced main table requires five
seeds; extra TimesNet, 48-hour and regional-transfer runs remain explicitly
labelled. No pilot result is used. Paired inference keeps missing slots on the
time grid and crosses common seed draws with common circular time blocks.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stable_shift_forecasting.blocked_inference import paired_block_bootstrap
from aggregate_benchmark_v3 import joint_bootstrap, holm

OUT=ROOT/'manuscript_v4/source_data'
SCENARIOS=['normal','missingness','shock','controlled_drift','anomaly']
BASELINES=['SeasonalNaive','WeeklyNaive','DLinear','PatchTST','TimesNet','iTransformer','TiDE']
SEEDS=[11,22,33,44,55]
R=5000
ROOTS=[ROOT/'results/benchmark_v3_formal',*[ROOT/'cloud_returned_20260909/results'/s for s in
       ['cloud_ecl_v3','cloud_ecl_h168_v3','cloud_ausgrid_v3','cloud_pjm_v3','benchmark_v3_transfer']]]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def primary_variant(m):return 'online' if m.startswith('StableShift') else 'frozen'
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,np.generic):v=v.item()
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    streams={};rows=[];availability=[];selections=[];source_files=[];data_records=[];protocols=[]
    for root in ROOTS:
        protocol=json.loads((root/'protocol.json').read_text());protocols.append(protocol)
        cfg=protocol['config']
        for folder in sorted(root.glob('*/h*')):
            if not (folder/'origins.json').exists():continue
            ds=folder.parent.name;h=int(folder.name[1:]);grid=json.loads((folder/'origins.json').read_text())['test']
            manifest=json.loads((folder.parent/'data_manifest.json').read_text())
            data_records.append(dict(dataset=ds,horizon=h,**{k:v for k,v in manifest.items() if k!='dataset'},
                                     training_origins=len(json.loads((folder/'origins.json').read_text()).get('train',[])),
                                     validation_origins=len(json.loads((folder/'origins.json').read_text()).get('validation',[]))))
            for selected in sorted(folder.glob('seed*/*/selected.json')):
                md=selected.parent;m=md.name;s=int(md.parent.name[4:]);sel=json.loads(selected.read_text())
                done=md/'completed.json'
                valid=done.exists();reason='complete' if valid else 'no completed stream record'
                variants=['online','no_update','no_filter','anchor_only'] if m=='StableShift' else [primary_variant(m)]
                cache=[]
                if valid:
                    completed=json.loads(done.read_text())
                    assert completed['signature']==sel['signature']==protocol['signature'],md
                    for var in variants:
                        for sc in SCENARIOS:
                            p=md/var/sc/'windows.csv';pred=p.with_name('predictions.npz')
                            if not p.exists() or not pred.exists():valid=False;reason='missing scenario artifact';break
                            d=pd.read_csv(p)
                            assert d.origin.tolist()==grid, f'Incomplete time grid: {p}'
                            assert np.array_equal(d.mae.notna(),d.scored),p
                            assert np.isfinite(d.loc[d.scored,['mae','mse','mase']]).all().all(),p
                            assert (d.loc[d.scored,['mae','mse','mase']]>=0).all().all(),p
                            with np.load(pred,allow_pickle=False) as z:
                                # Prediction key differs across early export versions.
                                arr=z['predictions'] if 'predictions' in z else z[z.files[0]]
                                assert len(arr)==len(grid) and np.isfinite(arr).all(),pred
                            summ=next((x for x in completed.get('summaries',[]) if x['scenario']==sc and x['variant']==var and x.get('subset','all')=='all'),{})
                            cache.append((var,sc,d,p,pred,summ))
                        if not valid:break
                availability.append(dict(dataset=ds,horizon=h,method=m,seed=s,complete=valid,reason=reason,
                                         root=str(root.relative_to(ROOT)),protocol_signature=protocol['signature']))
                if not valid:continue
                selections.append(dict(dataset=ds,horizon=h,method=m,seed=s,**{k:sel.get(k) for k in
                    ['best_validation_mae','best_epoch','epochs_run','lr','parameters','training_seconds','initial_hash','hit_epoch_budget']}))
                for var,sc,d,p,pred,summ in cache:
                    key=(ds,h,m,var,sc,s);assert key not in streams;streams[key]=d
                    q=d[d.scored]
                    rows.append(dict(dataset=ds,horizon=h,method=m,variant=var,scenario=sc,seed=s,
                        mae=q.mae.mean(),mse=q.mse.mean(),rmse=np.sqrt(q.mse.mean()),mase=q.mase.mean(),
                        scored_origins=len(q),deployment_origins=len(d),updates=int(d.updated.sum()),
                        natural_shock_origins=int((d.natural_shock & d.scored).sum()),
                        latency_median_ms=d.latency_ms.median(),latency_p95_ms=d.latency_ms.quantile(.95),
                        peak_gpu_mb=summ.get('peak_gpu_mb'),parameters=sel.get('parameters'),gpu=protocol.get('gpu','V100 (transfer run)'),
                        reconstruction_decrease_mean=(d.reconstruction_before-d.reconstruction_after).mean()))
                    source_files.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),prediction_path=pred.relative_to(ROOT).as_posix(),prediction_sha256=sha(pred)))
    df=pd.DataFrame(rows);df.to_csv(OUT/'seed_metrics.csv',index=False)
    av=pd.DataFrame(availability);av.to_csv(OUT/'run_coverage.csv',index=False)
    sd=pd.DataFrame(selections);sd.to_csv(OUT/'training_selection.csv',index=False)
    aggregates=[]
    for key,g in df.groupby(['dataset','horizon','method','variant','scenario'],sort=False):
        aggregates.append(dict(zip(['dataset','horizon','method','variant','scenario'],key),
            mae=g.mae.mean(),rmse=np.sqrt(g.mse.mean()),mase=g.mase.mean(),mae_seed_sd=g.mae.std(),
            seeds=len(g),seed_ids=','.join(map(str,sorted(g.seed))),scored_origins=int(g.scored_origins.iloc[0]),
            deployment_origins=int(g.deployment_origins.iloc[0]),updates_mean=g.updates.mean(),
            latency_median_ms=g.latency_median_ms.median(),parameters=g.parameters.iloc[0],
            peak_gpu_mb_max=g.peak_gpu_mb.max()))
    ag=pd.DataFrame(aggregates);ag.to_csv(OUT/'all_method_metrics.csv',index=False)
    main_cells=[('eia930',24),('eia930',168),('ecl',24),('ecl',168),('ausgrid',24),('ausgrid',168)]
    comparisons=[];ablations=[];validation=[];joint=[];mechanism=[]
    def array(ds,h,m,v,sc,seeds,metric):
        return np.stack([streams[(ds,h,m,v,sc,s)]['mse' if metric=='rmse' else metric].to_numpy() for s in seeds])
    for ds,h in main_cells:
        elig=[]
        for m in BASELINES:
            f=sd[(sd.dataset==ds)&(sd.horizon==h)&(sd.method==m)]
            if sorted(f.seed)==SEEDS:elig.append((m,f.best_validation_mae.mean()))
        baseline=min(elig,key=lambda x:x[1])[0]
        for m,val in elig:validation.append(dict(dataset=ds,horizon=h,method=m,validation_mae=val,selected=m==baseline))
        print(ds,h,'validation comparator',baseline,flush=True)
        for sc in SCENARIOS:
            for metric in ['mae','rmse','mase']:
                a=array(ds,h,baseline,'frozen',sc,SEEDS,metric);b=array(ds,h,'StableShift','online',sc,SEEDS,metric)
                for days in [7,14,28]:
                    res=paired_block_bootstrap(a,b,metric,int(np.ceil(24*days/h)),R)
                    comparisons.append(dict(dataset=ds,horizon=h,scenario=sc,baseline=baseline,metric=metric,block_days=days,**res))
                for var,m in [('no_update','StableShift'),('no_filter','StableShift'),('anchor_only','StableShift'),('no_stability','StableShiftNoStability')]:
                    v='online' if var=='no_stability' else var
                    if not all((ds,h,m,v,sc,s) in streams for s in SEEDS):continue
                    alt=array(ds,h,m,v,sc,SEEDS,metric)
                    res=paired_block_bootstrap(b,alt,metric,int(np.ceil(168/h)),R)
                    ablations.append(dict(dataset=ds,horizon=h,scenario=sc,variant=var,metric=metric,**res))
            rec=next(x for x in data_records if x['dataset']==ds and x['horizon']==h)
            start=pd.Timestamp(rec['validation_end_exclusive']);end=pd.Timestamp(rec['last_timestamp'])+pd.Timedelta(hours=1)
            f=streams[(ds,h,'StableShift','online',sc,11)]
            joint.append(dict(dataset=ds,horizon=h,scenario=sc,test_days=int(np.ceil((end-start)/pd.Timedelta(days=1))),
                 origin_days=np.floor((pd.DatetimeIndex(f.timestamp)-start)/pd.Timedelta(days=1)).astype(int).to_numpy(),
                 a={m:array(ds,h,baseline,'frozen',sc,SEEDS,m) for m in ['mae','rmse']},
                 b={m:array(ds,h,'StableShift','online',sc,SEEDS,m) for m in ['mae','rmse']}))
            for s in SEEDS:
                d=streams[(ds,h,'StableShift','online',sc,s)].copy();off=streams[(ds,h,'StableShift','no_update',sc,s)]
                d['mae_update_effect']=d.mae-off.mae
                mechanism.append(d)
    cmp=pd.DataFrame(comparisons);abl=pd.DataFrame(ablations)
    for days in [7,14,28]:
        for metrics in [['mae','rmse'],['mase']]:
            ix=(cmp.block_days==days)&cmp.metric.isin(metrics);cmp.loc[ix,'holm_p']=holm(cmp.loc[ix,'p_value'])
    abl['holm_p']=holm(abl.p_value)
    cmp.to_csv(OUT/'paired_comparisons.csv',index=False);abl.to_csv(OUT/'paired_ablations.csv',index=False)
    pd.DataFrame(validation).to_csv(OUT/'validation_comparators.csv',index=False)
    pd.concat(mechanism).to_csv(OUT/'online_telemetry.csv',index=False)
    result={scope:{m:joint_bootstrap(cells,m,R,7) for m in ['mae','rmse']} for scope,cells in
            [('chronological',[c for c in joint if c['scenario']=='normal']),('all_conditions',joint)]}
    metadata=dict(resamples=R,bootstrap='Paired crossed seed and circular time-block bootstrap',
                  main_cells=main_cells,comparison_direction='StableShift minus validation-selected available comparator',
                  ablation_direction='Removed component minus full StableShift',aggregate=result,
                  legacy_pilot_used=False,main_source_rule='Five completed seeds per method and full scheduled streams',
                  data=data_records,protocols=protocols,sources=source_files)
    (OUT/'analysis_manifest.json').write_text(json.dumps(clean(metadata),indent=2,allow_nan=False),encoding='utf-8')
    print(cmp[(cmp.scenario=='normal')&(cmp.metric=='mae')&(cmp.block_days==7)][['dataset','horizon','baseline','relative_difference','relative_ci_low','relative_ci_high']].to_string(index=False),flush=True)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
