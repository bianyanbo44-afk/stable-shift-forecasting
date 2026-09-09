"""Create tables and secondary analyses from saved forecast streams only."""
from pathlib import Path
import sys,json,re,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stable_shift_forecasting.blocked_inference import paired_block_bootstrap
O=ROOT/'manuscript_v4'; S=O/'source_data'; T=O/'tables'
T.mkdir(exist_ok=True)
A=pd.read_csv(S/'all_method_metrics.csv');C=pd.read_csv(S/'paired_comparisons.csv');B=pd.read_csv(S/'paired_ablations.csv')
SE=pd.read_csv(S/'seed_metrics.csv');SEL=pd.read_csv(S/'training_selection.csv')
M=json.loads((S/'analysis_manifest.json').read_text(encoding='utf-8'))
cells=[('eia930',24),('eia930',168),('ecl',24),('ecl',168),('ausgrid',24),('ausgrid',168)]
names={'eia930':'EIA-930','ecl':'ECL','ausgrid':'Ausgrid','eia930_ERCO':'ERCO transfer','eia930_SWPP':'SWPP transfer','pjm':'PJM archival'}
scnames={'normal':'Chronological','missingness':'Missing input','shock':'Recent amplitude','controlled_drift':'Affine offset','anomaly':'Block anomaly'}
methods=['SeasonalNaive','WeeklyNaive','DLinear','PatchTST','TimesNet','iTransformer','TiDE','StableShift']
mn={'SeasonalNaive':'Daily persistence','WeeklyNaive':'Weekly persistence','StableShiftNoStability':'Without representation penalties'}
def label(m):return mn.get(m,m)
def num(v,ds):return f'{v:.0f}' if ds.startswith('eia') or ds=='pjm' else (f'{v:.2f}' if ds=='ecl' else f'{v:.3f}')
def fci(row,relative=True):
    a,b,c=[row[k] for k in (['relative_difference','relative_ci_low','relative_ci_high'] if relative else ['difference','ci_low','ci_high'])]
    fac=100 if relative else 1
    return f'{a*fac:+.2f} [{b*fac:+.2f}, {c*fac:+.2f}]'
def tab(path,head,rows,cols,environment='tabular',size='\\small'):
    text='{\n'+size+'\n\\begin{'+environment+'}{'+cols+'}\n\\toprule\n'+head+' \\\\\n\\midrule\n'
    if environment=='longtable': text+='\\endhead\n'
    text+='\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)+'\n\\bottomrule\n\\end{'+environment+'}\n}\n'
    (T/path).write_text(text,encoding='utf-8')

# A balanced table: completed five-seed methods only.
rows=[]
for method in methods:
    r=[label(method)]
    for ds,h in cells:
        q=A[(A.dataset==ds)&(A.horizon==h)&(A.method==method)&(A.scenario=='normal')&A.variant.isin(['online','frozen'])&(A.seeds==5)]
        if q.empty:r.append('---');continue
        q=q.iloc[0];ref=A[(A.dataset==ds)&(A.horizon==h)&(A.scenario=='normal')&A.variant.isin(['online','frozen'])&(A.seeds==5)&A.method.isin(methods)]
        vals=[]
        for metric in ['mae','rmse']:
            v=num(q[metric],ds)
            if q[metric]==ref[metric].min():v='\\textbf{'+v+'}'
            vals.append(v)
        r.append(' / '.join(vals))
    rows.append(r)
tab('main_accuracy.tex','Method & EIA 24 h & EIA 168 h & ECL 24 h & ECL 168 h & Ausgrid 24 h & Ausgrid 168 h',rows,'lrrrrrr',size='\\scriptsize')

rows=[]
for ds,h in cells:
    q=C[(C.dataset==ds)&(C.horizon==h)&(C.scenario=='normal')&(C.block_days==7)]
    mae=q[q.metric=='mae'].iloc[0];rmse=q[q.metric=='rmse'].iloc[0]
    rows.append([names[ds],h,label(mae.baseline),fci(mae),fci(rmse),f'{mae.holm_p:.3f}'])
tab('primary_effects.tex','Data & $H$ & Comparator & MAE change (\\%) & RMSE change (\\%) & $p_{\\rm H}$ (MAE)',rows,'lllp{3.65cm}p{3.65cm}r',size='\\footnotesize')

rows=[]
for ds,h in cells:
    for sc in scnames:
        q=C[(C.dataset==ds)&(C.horizon==h)&(C.scenario==sc)&(C.block_days==7)&(C.metric=='mae')].iloc[0]
        rows.append([names[ds],h,scnames[sc],num(q.baseline_error,ds),num(q.proposed_error,ds),fci(q),f'{q.holm_p:.3f}'])
tab('stress_effects.tex','Data & $H$ & Input condition & Base MAE & SS MAE & Change (\\%) [95\\% CI] & $p_{\\rm H}$',rows,'lllr rp{3.65cm}r'.replace(' ',''),'longtable','\\footnotesize')

rows=[]
for ds,h in cells:
    rec=next(x for x in M['data'] if x['dataset']==ds and x['horizon']==h)
    q=SE[(SE.dataset==ds)&(SE.horizon==h)&(SE.method=='StableShift')&(SE.variant=='online')&(SE.scenario=='normal')].iloc[0]
    rows.append([names[ds],h,rec['training_origins'],rec['validation_origins'],int(q.deployment_origins),int(q.scored_origins),int(q.natural_shock_origins)])
tab('coverage.tex','Data & $H$ & Training & Validation & Deployed & Scored & Natural shock',rows,'llrrrrr')

rows=[]
for ds,h in cells:
    for v,lab in [('anchor_only','Anchor only'),('no_update','Online update off'),('no_filter','Input filter off'),('no_stability','Representation penalties off')]:
        for sc in scnames:
            q=B[(B.dataset==ds)&(B.horizon==h)&(B.scenario==sc)&(B.metric=='mae')&(B.variant==v)]
            if q.empty:continue
            q=q.iloc[0]
            digits=4 if v=='no_update' else 2
            effect=f'{q.relative_difference*100:+.{digits}f} [{q.relative_ci_low*100:+.{digits}f}, {q.relative_ci_high*100:+.{digits}f}]'
            rows.append([names[ds],h,lab,scnames[sc],effect,f'{q.holm_p:.3f}'])
tab('ablations.tex','Data & $H$ & Intervention & Condition & MAE change (\\%) [95\\% CI] & $p_{\\rm H}$',rows,'lllp{2.5cm}p{4.1cm}r','longtable','\\footnotesize')

rows=[]
for q in A.itertuples():
    if q.scenario!='normal' or q.variant not in ['online','frozen']:continue
    if (q.dataset,q.horizon) in cells and q.seeds==5:continue
    rows.append([names[q.dataset],q.horizon,label(q.method),q.seeds,num(q.mae,q.dataset),num(q.rmse,q.dataset),f'{q.mase:.3f}'])
tab('secondary_accuracy.tex','Data & $H$ & Method & Seeds & MAE & RMSE & MASE',rows,'llp{5.2cm}rrrr','longtable','\\footnotesize')

rows=[]
for ds,h in cells:
    for method in methods:
        q=A[(A.dataset==ds)&(A.horizon==h)&(A.method==method)&(A.scenario=='normal')&A.variant.isin(['online','frozen'])&(A.seeds==5)]
        if q.empty:continue
        q=q.iloc[0]
        rows.append([names[ds],h,label(method),f'{q.mase:.3f}',f'{int(q.parameters):,}',f'{q.latency_median_ms:.3f}',f'{q.peak_gpu_mb_max:.2f}'])
tab('cost_and_mase.tex','Data & $H$ & Method & MASE & Parameters & Median ms & Peak MiB',rows,'llp{3.4cm}rrrr','longtable','\\footnotesize')

# Three-seed region transfer: same paired stream design, reported separately.
tr=[];unc=[]
root=ROOT/'cloud_returned_20260909/results/benchmark_v3_transfer'
for ds in ['eia930_ERCO','eia930_SWPP']:
    subset=SEL[(SEL.dataset==ds)&(SEL.horizon==168)&SEL.method.isin(methods[:-1])]
    comparator=subset.groupby('method').best_validation_mae.mean().idxmin()
    for sc in scnames:
        def stream(m):
            v='online' if m=='StableShift' else 'frozen'
            return np.stack([pd.read_csv(root/ds/'h168'/f'seed{s}'/m/v/sc/'windows.csv').mse.to_numpy() for s in [11,22,33]])
        def mae(m):
            v='online' if m=='StableShift' else 'frozen'
            return np.stack([pd.read_csv(root/ds/'h168'/f'seed{s}'/m/v/sc/'windows.csv').mae.to_numpy() for s in [11,22,33]])
        for metric,fn in [('mae',mae),('rmse',stream)]:
            res=paired_block_bootstrap(fn(comparator),fn('StableShift'),metric,1,5000)
            tr.append(dict(dataset=ds,horizon=168,scenario=sc,baseline=comparator,metric=metric,**res))
        if sc=='normal':
            for method in ['SeasonalNaive','WeeklyNaive','DLinear','PatchTST','StableShift']:
                x=mae(method);rng=np.random.default_rng(2026);means=[]
                for _ in range(5000):
                    ss=rng.integers(0,3,size=3);ii=rng.integers(0,x.shape[1],size=x.shape[1]);means.append(np.nanmean(x[ss[:,None],ii]))
                unc.append(dict(dataset=ds,method=method,mean=np.nanmean(x),ci_low=np.quantile(means,.025),ci_high=np.quantile(means,.975),seeds=3,block_hours=168))
pd.DataFrame(tr).to_csv(S/'transfer_comparisons.csv',index=False)
pd.DataFrame(unc).to_csv(S/'transfer_absolute_intervals.csv',index=False)
rows=[[names[q['dataset']],scnames[q['scenario']],q['metric'].upper(),label(q['baseline']),fci(q)] for q in tr]
tab('transfer_effects.tex','Region & Condition & Metric & Comparator & Change (\\%) [95\\% CI]',rows,'lp{3.1cm}llp{4.2cm}','longtable','\\footnotesize')

# Save the displayed relative degradation ratios as Source Data.
rr=[]
for q in A.itertuples():
    if (q.dataset,q.horizon) not in cells or q.seeds!=5 or q.variant not in ['online','frozen']:continue
    base=A[(A.dataset==q.dataset)&(A.horizon==q.horizon)&(A.method==q.method)&(A.variant==q.variant)&(A.scenario=='normal')].iloc[0]
    rr.append(dict(dataset=q.dataset,horizon=q.horizon,method=q.method,scenario=q.scenario,mae_ratio=q.mae/base.mae,rmse_ratio=q.rmse/base.rmse))
pd.DataFrame(rr).to_csv(S/'degradation_ratios.csv',index=False)

# Registry fallbacks are tied to the existing dated verification records.
records=json.loads((S/'reference_verification.json').read_text(encoding='utf-8'))
cache=json.loads((ROOT/'literature/citation_verification_2026-09-08.json').read_text(encoding='utf-8'))['records']
lookup={x['doi'].lower():x for x in cache}
fresh=json.loads((ROOT/'literature/crossref_2026_verified.json').read_text(encoding='utf-8'))
if isinstance(fresh,dict): fresh=fresh['records']
if isinstance(fresh,list): lookup.update({x['doi'].lower():x for x in fresh if 'doi' in x})
for r in records:
    if r['status']=='live_lookup_unavailable' and r['doi'].lower() in lookup:
        r['cached_record']=lookup[r['doi'].lower()];r['status']='verified_cached_registry';r['cache_source']='literature/citation_verification_2026-09-08.json or crossref_2026_verified.json'
(S/'reference_verification.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')

# Distinguish the source variants instead of silently merging their digests.
snap=ROOT/'results/benchmark_v3_formal/source_snapshot'
other=ROOT/'cloud_returned_20260909/results/cloud_ausgrid_v3/source_snapshot'
diff=[]
for p in snap.rglob('*.py'):
    r=p.relative_to(snap);q=other/r
    if q.exists():
        a=p.read_text(encoding='utf-8').replace('\r\n','\n');b=q.read_text(encoding='utf-8').replace('\r\n','\n')
        if a!=b:diff.append(str(r))
(S/'source_snapshot_comparison.json').write_text(json.dumps({'different_text_files':diff,'local_digest':M['protocols'][0]['source_digest'],'cloud_digest':M['protocols'][1]['source_digest'],'interpretation':'Raw source digests retained separately; listed files differ after newline normalization.'},indent=2),encoding='utf-8')
print('Tables written:',len(list(T.glob('*.tex'))))
print('Transfer comparators:',pd.DataFrame(tr).groupby('dataset').baseline.first().to_dict())
print('Source snapshot differences:',diff)
print('Reference status:',pd.Series([r['status'] for r in records]).value_counts().to_dict())
