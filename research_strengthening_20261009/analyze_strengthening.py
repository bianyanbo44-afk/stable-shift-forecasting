"""Reconstruct all outcomes and run the predeclared paired inference."""
from pathlib import Path
import argparse,json,zlib,time
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
DEST=OUT/'analysis';DEST.mkdir(exist_ok=True)
P=json.loads((OUT/'protocol.json').read_text());AP=json.loads((OUT/'analysis_protocol.json').read_text())
SEEDS=P['seeds'];B=AP['bootstrap_draws']
PRIMARY=['frozen','cosa_expanded','nlms_expanded','query_expanded','history_clip','nlms_matched_new','history_matched_new']

def holm(p):
    values=np.array(p);order=np.argsort(values);out=np.empty_like(values)
    out[order]=np.minimum(1,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return out

def collect(phase):
    expected={'test':60,'independent':20,'withheld':40}[phase]
    done=sorted((OUT/phase).glob('*/h*/seed*/*/completed.json'))
    assert len(done)==expected,(phase,len(done),expected)
    frames=[];runs=[];geometry=[]
    for marker in done:
        folder=marker.parent;ds=folder.parents[2].name;h=int(folder.parents[1].name[1:]);seed=int(folder.parent.name[4:]);base=folder.name
        frozen=pd.read_csv(folder/'frozen/windows.csv')
        cosa=pd.read_csv(folder/'cosa_expanded/windows.csv')
        for scalar in [frozen,cosa]:
            if 'mase_observed_count' not in scalar:scalar['mase_observed_count']=scalar.observed_count
        assert np.array_equal(frozen.origin,cosa.origin) and np.array_equal(frozen.observed_count,cosa.observed_count)
        mapped=json.loads((folder/'method_mapping.json').read_text())
        with np.load(folder/'operators.npz') as z:arrays={k:z[k].copy() for k in z.files}
        records=json.loads((folder/'operators.json').read_text())['metrics']
        assert np.array_equal(frozen.origin,arrays['origins'])
        assert np.array_equal(frozen.observed_count,arrays['observed_count'])
        methods={'frozen':frozen,'cosa_expanded':cosa}
        for name,index in mapped.items():
            if index<0:frame=frozen.copy()
            else:
                frame=pd.DataFrame({'origin':arrays['origins'],'timestamp':arrays['timestamps'].astype(str),'observed_count':arrays['observed_count'],'mase_observed_count':arrays.get('mase_observed_count',arrays['observed_count']),'absolute_error_sum':arrays['errors'][index,:,0],'squared_error_sum':arrays['errors'][index,:,1],'scaled_error_sum':arrays['errors'][index,:,2],'max_update_label_index':arrays['max_update_label_index']})
                geo=arrays['geometry'][index]
                g=pd.DataFrame(geo,columns=['before_mse','after_mse','raw_query_rms','query_rms','query_cap','historical_cap','shrink'])
                g['update_number']=np.arange(len(g))
                for key,value in [('phase',phase),('dataset',ds),('horizon',h),('seed',seed),('base',base),('method',name)]:g[key]=value
                geometry.append(g)
            methods[name]=frame
        for method,frame in methods.items():
            assert (frame.max_update_label_index<frame.origin).all()
            count=frame.observed_count.sum()
            metric={'mae':float(frame.absolute_error_sum.sum()/count),'rmse':float(np.sqrt(frame.squared_error_sum.sum()/count)),'mase':float(frame.scaled_error_sum.sum()/frame.mase_observed_count.sum())}
            runs.append({'phase':phase,'dataset':ds,'horizon':h,'seed':seed,'base':base,'method':method,**metric,'observed_entries':int(count),'mase_observed_entries':int(frame.mase_observed_count.sum())})
            for key,value in [('phase',phase),('dataset',ds),('horizon',h),('seed',seed),('base',base),('method',method)]:frame[key]=value
            frames.append(frame)
    windows=pd.concat(frames,ignore_index=True)
    assert windows.groupby(['phase','dataset','horizon','seed','base','origin']).observed_count.nunique().max()==1
    windows.to_csv(DEST/f'{phase}_forecast_windows.csv',index=False)
    pd.DataFrame(runs).to_csv(DEST/f'{phase}_run_metrics.csv',index=False)
    pd.concat(geometry,ignore_index=True).to_csv(DEST/f'{phase}_update_geometry.csv',index=False)
    return windows

def draws(N,block,seed):
    rng=np.random.default_rng(seed);sw=rng.multinomial(len(SEEDS),np.ones(len(SEEDS))/len(SEEDS),size=B).astype(float)
    starts=rng.integers(0,N,size=(B,int(np.ceil(N/block))))
    indices=((starts[:,:,None]+np.arange(block))%N).reshape(B,-1)[:,:N]
    tw=np.zeros((B,N),np.float64)
    for i in range(B):tw[i]=np.bincount(indices[i],minlength=N)
    return sw,tw

def analyze(phase):
    t=time.monotonic();windows=collect(phase)
    keys=['phase','dataset','horizon','base','method']
    totals=windows.groupby(keys)[['observed_count','mase_observed_count','absolute_error_sum','squared_error_sum','scaled_error_sum']].sum().reset_index()
    totals['mae']=totals.absolute_error_sum/totals.observed_count
    totals['rmse']=np.sqrt(totals.squared_error_sum/totals.observed_count)
    totals['mase']=totals.scaled_error_sum/totals.mase_observed_count
    assert (totals.rmse+1e-10>=totals.mae).all()
    totals.to_csv(DEST/f'{phase}_aggregate_metrics.csv',index=False)
    paired=[];macro=[]
    families=[('primary',AP['primary']['baselines']),('matched_mechanism',AP['matched_mechanism']['baselines']),('independently_tuned_history',['history_clip'])]
    metrics={'mae':'absolute_error_sum','rmse':'squared_error_sum','mase':'scaled_error_sum'}
    for block in AP['block_days']:
        effects={}
        for ds,dsframe in windows[windows.method.isin(PRIMARY)].groupby('dataset'):
            grid=np.sort(dsframe.origin.unique());positions={int(o):i for i,o in enumerate(grid)};seedids={s:i for i,s in enumerate(SEEDS)}
            sw,tw=draws(len(grid),block,zlib.crc32(f'{phase}:{ds}:{block}'.encode()))
            for (h,base),frame in dsframe.groupby(['horizon','base']):
                arrays={}
                for method,rows in frame.groupby('method'):
                    count=np.zeros((len(SEEDS),len(grid)));mcount=count.copy();nums={m:count.copy() for m in metrics}
                    ix=([seedids[int(s)] for s in rows.seed],[positions[int(o)] for o in rows.origin])
                    count[ix]=rows.observed_count
                    mcount[ix]=rows.mase_observed_count
                    for metric,col in metrics.items():nums[metric][ix]=rows[col]
                    denominator=((sw@count)*tw).sum(1);assert (denominator>0).all()
                    values={}
                    for metric,num in nums.items():
                        used_count=mcount if metric=='mase' else count
                        used_denominator=((sw@used_count)*tw).sum(1)
                        assert (used_denominator>0).all()
                        point=num.sum()/used_count.sum();rep=((sw@num)*tw).sum(1)/used_denominator
                        if metric=='rmse':point=np.sqrt(point);rep=np.sqrt(rep)
                        values[metric]=(point,rep)
                    arrays[method]=values
                for family,baselines in families:
                    for baseline in baselines:
                        for metric in metrics:
                            ref,reps=arrays[baseline][metric];q,qreps=arrays['query_expanded'][metric]
                            diff=q-ref;drep=qreps-reps;relative=q/ref-1;rrep=qreps/reps-1
                            row={'phase':phase,'dataset':ds,'horizon':h,'base':base,'family':family,'baseline':baseline,'proposed':'query_expanded','metric':metric,'block_days':block,'baseline_error':float(ref),'proposed_error':float(q),'difference':float(diff),'ci_low':float(np.quantile(drep,.025)),'ci_high':float(np.quantile(drep,.975)),'relative_difference':float(relative),'relative_ci_low':float(np.quantile(rrep,.025)),'relative_ci_high':float(np.quantile(rrep,.975)),'p_value':float((1+np.count_nonzero(np.abs(drep-diff)>=abs(diff)))/(B+1))}
                            paired.append(row)
                            effects.setdefault((family,baseline,metric),[]).append((relative,rrep))
        for (family,baseline,metric),rows in effects.items():
            point=np.mean([r[0] for r in rows]);rep=np.mean(np.stack([r[1] for r in rows]),0)
            macro.append({'phase':phase,'family':family,'baseline':baseline,'metric':metric,'block_days':block,'cells':len(rows),'relative_difference':point,'ci_low':np.quantile(rep,.025),'ci_high':np.quantile(rep,.975)})
    pair=pd.DataFrame(paired);pair['metric_family']=np.where(pair.metric=='mase','mase','mae_rmse')
    for _,idx in pair.groupby(['family','metric_family','block_days']).groups.items():
        pair.loc[idx,'holm_p']=holm(pair.loc[idx,'p_value']);pair.loc[idx,'family_size']=len(idx)
    for family in ['primary','matched_mechanism','independently_tuned_history']:
        assert int(pair[(pair.family==family)&(pair.metric_family=='mae_rmse')&(pair.block_days==7)].family_size.iloc[0])==AP[family]['holm_tests'][phase]
    pair.to_csv(DEST/f'{phase}_paired_comparisons.csv',index=False)
    pd.DataFrame(macro).to_csv(DEST/f'{phase}_macro_comparisons.csv',index=False)
    geo=pd.read_csv(DEST/f'{phase}_update_geometry.csv')
    summaries=[]
    for key,frame in geo.groupby(['dataset','horizon','base','method']):
        summaries.append(dict(zip(['dataset','horizon','base','method'],key))|{'phase':phase,'updates':len(frame),'clipped_fraction':float((frame.shrink<1-1e-6).mean()),'query_rms_median':float(frame.query_rms.median()),'query_rms_p95':float(frame.query_rms.quantile(.95)),'query_rms_p99':float(frame.query_rms.quantile(.99)),'query_rms_max':float(frame.query_rms.max())})
    pd.DataFrame(summaries).to_csv(DEST/f'{phase}_geometry_summary.csv',index=False)
    print(json.dumps({'phase':phase,'windows':len(windows),'aggregate_rows':len(totals),'seconds':time.monotonic()-t,'macro_mae_7day':[r for r in macro if r['metric']=='mae' and r['block_days']==7]},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['test','independent','withheld','all'],default='all');args=parser.parse_args()
    for phase in (['test','independent','withheld'] if args.phase=='all' else [args.phase]):analyze(phase)
