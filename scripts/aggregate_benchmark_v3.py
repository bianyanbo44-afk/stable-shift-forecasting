"""Build source-data tables from complete, paired deployment streams.

Validation selects one baseline per dataset/horizon across the five seeds.
Single-cell inference uses paired circular time blocks crossed with seed draws.
Macro and worst-case inference jointly resamples calendar-day blocks within a
dataset so all horizons/scenarios share the same temporal resampling weights.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stable_shift_forecasting.blocked_inference import paired_block_bootstrap

BASELINES=['SeasonalNaive','WeeklyNaive','DLinear','PatchTST','TimesNet','iTransformer','TiDE']
METRICS={'mae':'mae','rmse':'mse','mase':'mase'}


def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);adjusted=np.empty_like(p)
    adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return adjusted


def read_stream(folder,method,variant,scenario,seeds,grid,subset='all'):
    arrays={key:[] for key in METRICS};frames=[]
    for seed in seeds:
        path=folder/f'seed{seed}'/method/variant/scenario/'windows.csv'
        df=pd.read_csv(path).set_index('origin')
        if df.index.has_duplicates or not df.index.is_monotonic_increasing:
            raise ValueError(f'Non-chronological/duplicate origins: {path}')
        if not np.array_equal(df.index.to_numpy(),grid):
            raise ValueError(f'Deployment stream omits scheduled origins: {path}')
        if subset=='natural_shock':
            df.loc[~df.natural_shock, list(METRICS.values())]=np.nan
        for metric,column in METRICS.items(): arrays[metric].append(df[column].to_numpy())
        frames.append(df)
    return {key:np.stack(value) for key,value in arrays.items()},frames


def joint_bootstrap(cells,metric,resamples,block_days,seed=2026):
    """Equal-cell relative error and maximum relative error, shared time draws.

    Dataset scales are removed by within-cell ratios. The same seed sample is
    used across all cells, and the same day-block weights across every horizon
    and scenario from each dataset. Ratios are computed after pooling errors.
    """
    rng=np.random.default_rng(seed)
    datasets={c['dataset']:c['test_days'] for c in cells}
    S=cells[0]['a'][metric].shape[0]
    observed=[];draws=[]
    for c in cells:
        a,b=np.nanmean(c['a'][metric]),np.nanmean(c['b'][metric])
        if metric=='rmse': a,b=np.sqrt(a),np.sqrt(b)
        observed.append(b/a-1)
    attempts=0
    while len(draws)<resamples:
        attempts+=1
        if attempts>resamples*20: raise ValueError('Too few scored blocks for joint bootstrap')
        seeds=rng.integers(0,S,size=S)
        weights={}
        for dataset,N in datasets.items():
            length=min(block_days,N)
            indices=((rng.integers(0,N,size=int(np.ceil(N/length)))[:,None]+np.arange(length))%N).ravel()[:N]
            weights[dataset]=np.bincount(indices,minlength=N)
        effects=[]
        for c in cells:
            a,b=c['a'][metric][seeds],c['b'][metric][seeds]
            w=weights[c['dataset']][c['origin_days']][None,:]
            valid=np.isfinite(a)
            denominator=(w*valid).sum()
            if denominator==0: break
            va=(w*np.nan_to_num(a)).sum()/denominator
            vb=(w*np.nan_to_num(b)).sum()/denominator
            if va<=0: break
            effects.append(np.sqrt(vb/va)-1 if metric=='rmse' else vb/va-1)
        if len(effects)==len(cells): draws.append(effects)
    draws=np.asarray(draws);observed=np.asarray(observed)
    output={}
    for name,summary,point in [('macro',draws.mean(1),observed.mean()),('worst_case',draws.max(1),observed.max())]:
        output[name]={'relative_difference':float(point),
            'relative_ci_low':float(np.quantile(summary,.025)),
            'relative_ci_high':float(np.quantile(summary,.975)),
            'cells':len(cells),'resamples':resamples,'block_days':block_days}
    return output


def aggregate(root,destination,resamples,allow_partial=False,additional_roots=()):
    protocol=json.loads((root/'protocol.json').read_text())
    config=protocol['config'];seeds=config['seeds'];summaries=[];comparisons=[];ablations=[]
    selections=[];incomplete=[];cells=[];sources=[]
    roots=[root,*additional_roots]
    for candidate in additional_roots:
        other=json.loads((candidate/'protocol.json').read_text())
        if other['source_digest']!=protocol['source_digest']:
            raise ValueError('Combined benchmarks must share model/training source')
        for key in ['seeds','horizons','methods','stable_model','learning_rates','epochs','patience']:
            if other['config'][key]!=config[key]:raise ValueError(f'Mismatched protocol field {key}')
    for dataset in config['datasets']:
        for horizon in config['horizons']:
            candidates=[r for r in roots if (r/dataset/f'h{horizon}'/'origins.json').exists()]
            if len(candidates)>1:raise ValueError(f'Duplicate dataset/horizon result roots: {dataset} H{horizon}')
            dataset_root=candidates[0] if candidates else root
            folder=dataset_root/dataset/f'h{horizon}'
            required=BASELINES+['StableShift','StableShiftNoStability']
            missing=[f'{dataset}/h{horizon}/seed{s}/{m}' for s in seeds for m in required
                     if not (folder/f'seed{s}'/m/'completed.json').exists()]
            if missing:
                incomplete.extend(missing);continue
            origin_record=json.loads((folder/'origins.json').read_text())
            grid=np.asarray(origin_record['test'])
            manifest=json.loads((dataset_root/dataset/'data_manifest.json').read_text())
            start=pd.Timestamp(manifest['validation_end_exclusive'])
            end=pd.Timestamp(manifest['last_timestamp'])+pd.Timedelta(hours=1)
            test_days=int(np.ceil((end-start)/pd.Timedelta(days=1)))
            validation={}
            for method in BASELINES:
                entries=[json.loads((folder/f'seed{s}'/method/'selected.json').read_text()) for s in seeds]
                validation[method]=float(np.mean([e['best_validation_mae'] for e in entries]))
            baseline=min(validation,key=validation.get)
            for method,score in validation.items():
                selections.append({'dataset':dataset,'horizon':horizon,'method':method,
                                   'validation_mae':score,'selected':method==baseline})
            for scenario in config['scenarios']:
                for subset in ['all','natural_shock'] if scenario=='normal' else ['all']:
                    values,frames=read_stream(folder,'StableShift','online',scenario,seeds,grid,subset)
                    if not np.isfinite(values['mae']).any():continue
                    reference,_=read_stream(folder,baseline,'frozen',scenario,seeds,grid,subset)
                    days=np.floor((pd.DatetimeIndex(frames[0].timestamp)-start)/pd.Timedelta(days=1)).astype(int)
                    cells.append({'dataset':dataset,'horizon':horizon,'scenario':scenario,'subset':subset,
                                  'a':reference,'b':values,'origin_days':np.asarray(days),'test_days':test_days})
                    common={'dataset':dataset,'horizon':horizon,'scenario':scenario,'subset':subset,
                            'baseline':baseline,'proposed':'StableShift'}
                    for metric in METRICS:
                        for days_block in [7,14,28]:
                            result=paired_block_bootstrap(reference[metric],values[metric],metric,
                                block_length=int(np.ceil(24*days_block/horizon)),resamples=resamples)
                            comparisons.append({**common,'metric':metric,'block_days':days_block,**result})
                    for variant in ['no_update','no_filter','anchor_only','no_stability_loss']:
                        method='StableShiftNoStability' if variant=='no_stability_loss' else 'StableShift'
                        load_variant='online' if variant=='no_stability_loss' else variant
                        alternative,_=read_stream(folder,method,load_variant,scenario,seeds,grid,subset)
                        for metric in METRICS:
                            result=paired_block_bootstrap(values[metric],alternative[metric],metric,
                                block_length=int(np.ceil(168/horizon)),resamples=resamples)
                            ablations.append({**common,'baseline':'StableShift','proposed':variant,'metric':metric,**result})
                # Full methods table is derived from window errors, never rounded
                # summaries. RMSE is the square root of pooled squared errors.
                for method in required:
                    variant='online' if method.startswith('StableShift') else 'frozen'
                    errors,frames=read_stream(folder,method,variant,scenario,seeds,grid)
                    summaries.append({'dataset':dataset,'horizon':horizon,'scenario':scenario,'method':method,
                        'mae':float(np.nanmean(errors['mae'])),
                        'rmse':float(np.sqrt(np.nanmean(errors['rmse']))),
                        'mase':float(np.nanmean(errors['mase'])),
                        'seeds':len(seeds),'scored_origins':int(np.isfinite(errors['mae'][0]).sum()),
                        'deployment_origins':len(grid),'updates_mean':float(np.mean([f.updated.sum() for f in frames])),
                        'latency_median_ms':float(pd.concat(frames).latency_ms.median())})
    if incomplete and not allow_partial:
        raise RuntimeError(f'{len(incomplete)} required method/seed cells incomplete. Use --allow-partial for a labelled progress export.')
    destination.mkdir(parents=True,exist_ok=True)
    if not comparisons:
        (destination/'status.json').write_text(json.dumps({'complete':False,'missing':incomplete},indent=2))
        print(f'No complete five-seed dataset/horizon cells; {len(incomplete)} runs pending.');return
    result=pd.DataFrame(comparisons)
    # Primary family includes all specified MAE and RMSE comparisons. Sensitivity
    # block lengths and secondary MASE comparisons have their own families.
    result['holm_p']=np.nan
    for days in [7,14,28]:
        for metrics in [['mae','rmse'],['mase']]:
            ix=(result.block_days==days)&result.metric.isin(metrics)
            result.loc[ix,'holm_p']=holm(result.loc[ix,'p_value'])
    ablation_frame=pd.DataFrame(ablations)
    ablation_frame['holm_p']=holm(ablation_frame.p_value)
    result.to_csv(destination/'paired_comparisons.csv',index=False)
    ablation_frame.to_csv(destination/'paired_ablations.csv',index=False)
    pd.DataFrame(summaries).to_csv(destination/'method_metrics.csv',index=False)
    pd.DataFrame(selections).to_csv(destination/'validation_selection.csv',index=False)
    aggregate_results={}
    for scope,subset in [('temporal',[c for c in cells if c['scenario']=='normal' and c['subset']=='all']),
                         ('all_evaluated_scenarios',cells)]:
        aggregate_results[scope]={m:joint_bootstrap(subset,m,resamples,7) for m in ['mae','rmse']}
    # Distinguish coverage from effect criteria so a partial run cannot be
    # mistaken for a fully established headline result.
    selected=aggregate_results['all_evaluated_scenarios']
    effect_pass=all(selected[m]['macro']['relative_ci_high']<0 and
                    selected[m]['worst_case']['relative_ci_high']<=.01 for m in ['mae','rmse'])
    component_criteria={}
    for variant in ['no_update','no_stability_loss']:
        subset=ablation_frame[(ablation_frame.proposed==variant)&ablation_frame.metric.isin(['mae','rmse']) &
                             ~((ablation_frame.scenario=='normal')&(ablation_frame['subset']=='all'))]
        grouped=subset.groupby(['dataset','horizon','scenario','subset'])
        degraded=grouped.apply(lambda x:bool((x.difference>0).any()),include_groups=False)
        component_criteria[variant]={'cells':len(degraded),'cells_degraded':int(degraded.sum()),
                                    'majority_degrade':bool(len(degraded)>0 and degraded.mean()>.5)}
    for candidate in roots:
        for path in sorted(candidate.glob('*/h*/seed*/*/*/*/windows.csv')):
            sources.append({'root':str(candidate),'path':str(path.relative_to(candidate)),
                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    report={'protocol_signature':protocol['signature'],'complete':not incomplete,
            'missing':incomplete,'evaluated_effect_criteria_pass':effect_pass,
            'component_criteria':component_criteria,
            'primary_claim_established':not incomplete and effect_pass and all(x['majority_degrade'] for x in component_criteria.values()),
            'aggregate':aggregate_results,'source_files':sources,
            'ablation_direction':'Positive difference means removing the component increased error.',
            'comparison_direction':'Negative difference means StableShift reduced error.'}
    (destination/'inference.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'complete':not incomplete,'complete_dataset_horizon_cells':len(selections)//len(BASELINES),
                      'effect_criteria_pass':effect_pass,'output':str(destination)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/benchmark_v3_formal')
    p.add_argument('--output',default='results/benchmark_v3_statistics');p.add_argument('--resamples',type=int,default=5000)
    p.add_argument('--allow-partial',action='store_true');p.add_argument('--additional-root',action='append',default=[]);a=p.parse_args()
    aggregate(ROOT/a.root,ROOT/a.output,a.resamples,a.allow_partial,[ROOT/r for r in a.additional_root])
