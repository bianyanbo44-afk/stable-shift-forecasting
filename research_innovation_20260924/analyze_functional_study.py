"""Paired seed/calendar-block inference for observed-error sum/count estimands."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/functional_calibration_20260924'
OUT=ROOT/'submission_kbs_functional_20260924/source_data';OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[11,22,33,44,55];METHODS=['frozen','cosa','cosa_scale_01','cosa_scale_10','query','nlms','nlms_matched']
METRICS={'mae':'absolute_error_sum','rmse':'squared_error_sum','mase':'scaled_error_sum'}
B=5000;START=time.monotonic()


def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty_like(p)
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out


def draws(S,N,block,seed):
    rng=np.random.default_rng(seed)
    sw=rng.multinomial(S,np.ones(S)/S,size=B).astype(float)
    starts=rng.integers(0,N,size=(B,int(np.ceil(N/block))))
    indices=((starts[:,:,None]+np.arange(block))%N).reshape(B,-1)[:,:N]
    tw=np.zeros((B,N),np.float64)
    for i in range(B):tw[i]=np.bincount(indices[i],minlength=N)
    return sw,tw


def ratio_sample(num,count,sw,tw,sqrt=False):
    denominator=((sw@count)*tw).sum(1)
    numerator=((sw@num)*tw).sum(1)
    assert np.all(denominator>0)
    value=numerator/denominator
    point=num.sum()/count.sum()
    return (np.sqrt(point),np.sqrt(value)) if sqrt else (point,value)


def contrast(a,ar,b,br):
    diff=b-a;rep=br-ar;rel=b/a-1;ratios=br/ar-1
    return {'baseline_error':float(a),'proposed_error':float(b),'difference':float(diff),
        'ci_low':float(np.quantile(rep,.025)),'ci_high':float(np.quantile(rep,.975)),
        'relative_difference':float(rel),'relative_ci_low':float(np.quantile(ratios,.025)),
        'relative_ci_high':float(np.quantile(ratios,.975)),
        'p_value':float((1+np.count_nonzero(np.abs(rep-diff)>=abs(diff)))/(B+1))}


allwindows=[];runmetrics=[];geometry=[];validation=[]
for phase,expected in [('test',60),('independent',20)]:
    folders=sorted((R/phase).glob('*/h*/seed*/*/completed.json'))
    assert len(folders)==expected,(phase,len(folders))
    for completed in folders:
        folder=completed.parent;ds=folder.parents[2].name;h=int(folder.parents[1].name[1:]);seed=int(folder.parent.name[4:]);base=folder.name
        counts=None
        for method in METHODS:
            p=folder/method
            w=pd.read_csv(p/'windows.csv')
            assert not w.origin.duplicated().any() and np.all(np.diff(w.origin)==24)
            assert (w.max_update_label_index<w.origin).all()
            if counts is None:counts=w[['origin','observed_count']].copy()
            else:assert counts.equals(w[['origin','observed_count']])
            m=json.loads((p/'metrics.json').read_text())
            for metric,col in METRICS.items():
                value=w[col].sum()/w.observed_count.sum()
                if metric=='rmse':value=np.sqrt(value)
                assert np.isclose(value,m[metric],rtol=1e-10,atol=1e-10),(phase,ds,h,seed,base,method,metric)
            for key,value in [('phase',phase),('dataset',ds),('horizon',h),('seed',seed),('base',base),('method',method)]:w[key]=value
            allwindows.append(w)
            runmetrics.append({'phase':phase,'dataset':ds,'horizon':h,'seed':seed,'base':base,'method':method,**m})
            if method=='query' and (p/'update_geometry.csv').exists():
                g=pd.read_csv(p/'update_geometry.csv')
                if len(g):
                    assert (g.after_mse<=g.before_mse+2e-6*np.maximum(1,g.before_mse)).all()
                    assert (g.query_rms<=g.cap+2e-6*np.maximum(1,g.cap)).all()
                    for key,value in [('phase',phase),('dataset',ds),('horizon',h),('seed',seed),('base',base)]:g[key]=value
                    geometry.append(g)
        validation.append({'phase':phase,'dataset':ds,'horizon':h,'seed':seed,'base':base,'all_masks_identical':True,'metrics_roundtrip':True,'causal_updates':True})

W=pd.concat(allwindows,ignore_index=True);W.to_csv(OUT/'forecast_windows.csv',index=False)
M=pd.DataFrame(runmetrics);M.to_csv(OUT/'run_metrics.csv',index=False)
G=pd.concat(geometry,ignore_index=True);G.to_csv(OUT/'query_update_geometry.csv',index=False)
pd.DataFrame(validation).to_csv(OUT/'run_validation.csv',index=False)

aggregates=[];paired=[];macro=[];complete=[];drawcache={}
for phase,pdf in W.groupby('phase'):
    arrays={};axes={}
    for ds,g in pdf.groupby('dataset'):
        axis=np.arange(g.origin.min(),g.origin.max()+24,24)
        axes[ds]=axis
        for (h,base,method),q in g.groupby(['horizon','base','method']):
            slot={}
            for col in list(METRICS.values())+['observed_count']:
                slot[col]=np.stack([q[q.seed==seed].set_index('origin')[col].reindex(axis,fill_value=0.).to_numpy(float) for seed in SEEDS])
            arrays[ds,h,base,method]=slot
            count=slot['observed_count'].sum()
            row={'phase':phase,'dataset':ds,'horizon':h,'base':base,'method':method,'observed_entries_per_seed':count/5}
            for metric,col in METRICS.items():
                value=slot[col].sum()/count
                row[metric]=float(np.sqrt(value) if metric=='rmse' else value)
            aggregates.append(row)
            full=q[q.complete_block]
            if len(full):
                complete.append({'phase':phase,'dataset':ds,'horizon':h,'base':base,'method':method,
                  'complete_mae':float(full.absolute_error_sum.sum()/full.observed_count.sum()),
                  'complete_rmse':float(np.sqrt(full.squared_error_sum.sum()/full.observed_count.sum())),
                  'complete_origins_per_seed':len(full)/5})
    for days in [7,14,28]:
        values={}
        for key,slot in arrays.items():
            ds,h,base,method=key
            dk=(phase,ds,days)
            if dk not in drawcache:
                seed=20260924+days+{'eia930':0,'ecl':100,'ausgrid':200}[ds]+(1000 if phase=='independent' else 0)
                drawcache[dk]=draws(5,len(axes[ds]),days,seed)
            sw,tw=drawcache[dk]
            for metric,col in METRICS.items():values[key+(metric,)]=ratio_sample(slot[col],slot['observed_count'],sw,tw,metric=='rmse')
        cellkeys=sorted(set((ds,h,base) for ds,h,base,_ in arrays))
        comparisons=[('main','frozen','query'),('main','cosa','query'),('main','nlms','query'),
                     ('constraint','nlms_matched','query'),('coordinate','cosa','cosa_scale_01'),('coordinate','cosa','cosa_scale_10')]
        for ds,h,base in cellkeys:
            for family,baseline,proposed in comparisons:
                for metric in METRICS:
                    a,ar=values[ds,h,base,baseline,metric];b,br=values[ds,h,base,proposed,metric]
                    paired.append({'phase':phase,'dataset':ds,'horizon':h,'base':base,'family':family,
                       'baseline':baseline,'proposed':proposed,'metric':metric,'block_days':days,**contrast(a,ar,b,br)})
        for baseline in ['frozen','cosa','nlms','nlms_matched']:
            for metric in METRICS:
                point=[];rep=[]
                for ds,h,base in cellkeys:
                    a,ar=values[ds,h,base,baseline,metric];b,br=values[ds,h,base,'query',metric]
                    point.append(b/a-1);rep.append(br/ar-1)
                point=np.array(point);rep=np.stack(rep)
                for estimand,pv,rv in [('mean',point.mean(),rep.mean(0)),('worst',point.max(),rep.max(0))]:
                    macro.append({'phase':phase,'baseline':baseline,'metric':metric,'block_days':days,'estimand':estimand,
                       'relative_difference':float(pv),'ci_low':float(np.quantile(rv,.025)),'ci_high':float(np.quantile(rv,.975)),
                       'cells':len(cellkeys)})

A=pd.DataFrame(aggregates);A.to_csv(OUT/'aggregate_metrics.csv',index=False)
P=pd.DataFrame(paired)
P['metric_family']=np.where(P.metric=='mase','mase','mae_rmse')
for key,indices in P.groupby(['phase','block_days','family','metric_family']).groups.items():
    P.loc[indices,'holm_p']=holm(P.loc[indices,'p_value'])
    P.loc[indices,'family_size']=len(indices)
P.to_csv(OUT/'paired_comparisons.csv',index=False)
pd.DataFrame(macro).to_csv(OUT/'macro_comparisons.csv',index=False)
pd.DataFrame(complete).to_csv(OUT/'complete_block_sensitivity.csv',index=False)

speed=[]
for key,g in M.groupby(['phase','dataset','horizon','seed','base']):
    q=g[g.method=='query'].iloc[0];c=g[g.method=='cosa'].iloc[0]
    speed.append(dict(zip(['phase','dataset','horizon','seed','base'],key),query_seconds=q.seconds,cosa_seconds=c.seconds,
        speedup=float(c.seconds/q.seconds),query_updates=int(q.updates),cosa_updates=int(c.updates)))
pd.DataFrame(speed).to_csv(OUT/'calibration_runtime.csv',index=False)
report={'runs_validated':len(validation),'test_base_cells':60,'independent_base_cells':20,
 'query_updates_audited':len(G),'all_query_descent_checks_pass':True,'all_query_bound_checks_pass':True,
 'constraint_active_fraction':float((G.shrink<1-1e-6).mean()),
 'max_query_ratio':float((G.query_rms/G.cap).max()),'paired_bootstrap_draws':B,
 'block_days':[7,14,28],'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'elapsed_seconds':time.monotonic()-START}
(OUT/'analysis_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(pd.DataFrame(macro).query("metric=='mae' and block_days==7").to_string(index=False),flush=True)
print(P.query("phase=='independent' and metric=='mae' and block_days==7 and family=='main'")[[
 'horizon','base','baseline','relative_difference','relative_ci_low','relative_ci_high','holm_p']].to_string(index=False),flush=True)
print(json.dumps(report,indent=2),flush=True)
