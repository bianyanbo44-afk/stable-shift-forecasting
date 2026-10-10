"""Audit complete evidence and derive manuscript tables without changing outcomes."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;A=HERE/'analysis';T=HERE/'tables';T.mkdir(exist_ok=True)
NAMES={'frozen':'Frozen','cosa_expanded':'COSA','nlms_expanded':'NLMS',
 'history_clip':'Historical clipping','nlms_matched_new':'Matched NLMS','history_matched_new':'Matched historical clipping'}
DS={'eia930':'EIA','ecl':'ECL','ausgrid':'Ausgrid'}
CELLS=[(ds,h,b) for ds in DS for h in [24,168] for b in ['DLinear','StableShift']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def tabular(headers,rows,colspec):
    return '\\begin{tabular}{'+colspec+'}\n\\toprule\n'+' & '.join(headers)+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(map(str,r))+'\\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n'

def interval(row,relative=False):
    point=row.relative_difference*100
    lo=row.relative_ci_low*100 if relative else row.ci_low*100
    hi=row.relative_ci_high*100 if relative else row.ci_high*100
    return f'${point:+.2f}$ [$ {lo:+.2f}, {hi:+.2f} $]'

def pformat(p):return '$<0.001$' if p<.001 else f'{p:.3f}'

def paired_table(phase,family,baselines):
    p=pd.read_csv(A/f'{phase}_paired_comparisons.csv')
    p=p[(p.family==family)&(p.metric=='mae')&(p.block_days==7)]
    lines=[]
    for ds,h,b in CELLS:
        if phase=='withheld' and ds=='eia930':continue
        if phase=='independent' and ds!='eia930':continue
        for ref in baselines:
            r=p[(p.dataset==ds)&(p.horizon==h)&(p.base==b)&(p.baseline==ref)].iloc[0]
            lines.append(f'{DS[ds]} & {h} & {b} & {NAMES[ref]} & {interval(r,True)} & {pformat(r.holm_p)}\\\\')
    phase_name={'test':'Original benchmark re-evaluation','independent':'EIA re-evaluation','withheld':'Unused-entity transfer'}[phase]
    family_name={'primary':'primary contrasts','matched_mechanism':'matched-step controls','independently_tuned_history':'independently tuned historical clipping'}[family]
    header='Dataset & $H$ & Source & Comparator & MAE difference (\\%) [95\\% CI] & Holm $p$\\\\\n\\midrule\n'
    out='\\begin{longtable}{lllp{.22\\textwidth}rp{.09\\textwidth}}\n\\caption{'+phase_name+': QNC '+family_name+'. Relative MAE differences with pointwise paired 95\\% intervals.}\\label{tab:'+phase+'_'+family+'}\\\\\n\\toprule\n'+header+'\\endfirsthead\n\\toprule\n'+header+'\\endhead\n'+'\n'.join(lines)+'\n\\bottomrule\n\\end{longtable}\n'
    (T/f'{phase}_{family}.tex').write_text(out,encoding='utf-8')

def audit_phase(phase):
    w=pd.read_csv(A/f'{phase}_forecast_windows.csv');r=pd.read_csv(A/f'{phase}_run_metrics.csv');g=pd.read_csv(A/f'{phase}_update_geometry.csv')
    assert (w.max_update_label_index<w.origin).all()
    assert w.groupby(['dataset','horizon','base','seed','origin']).observed_count.nunique().max()==1
    sums=w.groupby(['dataset','horizon','base','seed','method'])[['observed_count','mase_observed_count','absolute_error_sum','squared_error_sum','scaled_error_sum']].sum()
    for row in r.itertuples():
        num=sums.loc[(row.dataset,row.horizon,row.base,row.seed,row.method)]
        for key,value in [('mae',num.absolute_error_sum/num.observed_count),('rmse',np.sqrt(num.squared_error_sum/num.observed_count)),('mase',num.scaled_error_sum/num.mase_observed_count)]:
            assert np.isclose(value,getattr(row,key),rtol=1e-10,atol=1e-10)
    assert (g.after_mse<=g.before_mse+2e-6*np.maximum(1,g.before_mse)).all()
    q=g[g.method=='query_expanded']
    assert (q.query_rms<=q.query_cap+2e-6*np.maximum(1,q.query_cap)).all()
    summary={'source_runs':len(r[r.method=='frozen']),'forecast_rows':len(w),'mask_and_causal_checks':True,'metrics_recomputed':True,'active_qnc_updates':len(q),'qnc_clipped_updates':int((q.shrink<1-1e-6).sum()),'qnc_bound_violations':0,'operator_feedback_loss_violations':0,
      'qnc_dataset_updates':{ds:{'updates':len(part),'clipped':int((part.shrink<1-1e-6).sum())} for ds,part in q.groupby('dataset')}}
    methods=['frozen','cosa_expanded','nlms_expanded','query_expanded','history_clip']
    agg=pd.read_csv(A/f'{phase}_aggregate_metrics.csv')
    rows=[]
    for ds,h,b in CELLS:
        if phase=='withheld' and ds=='eia930':continue
        if phase=='independent' and ds!='eia930':continue
        sub=agg[(agg.dataset==ds)&(agg.horizon==h)&(agg.base==b)].set_index('method')
        units=0 if ds=='eia930' else 3
        rows.append([DS[ds],h,b,*[f'{sub.loc[m,"mae"]:,.{units}f}' for m in methods]])
    (T/f'{phase}_absolute_mae.tex').write_text(tabular(['Dataset','$H$','Source','Frozen','COSA','NLMS','QNC','History'],rows,'lllrrrrr'),encoding='utf-8')
    paired_table(phase,'primary',['frozen','cosa_expanded','nlms_expanded'])
    paired_table(phase,'matched_mechanism',['nlms_matched_new','history_matched_new'])
    paired_table(phase,'independently_tuned_history',['history_clip'])
    return summary

def selections():
    lock=json.loads((HERE/'selection_lock.json').read_text());assert len(lock['files'])==60
    rows=[]
    for rel,digest in lock['files'].items():
        file=HERE/rel;assert sha(file)==digest
        selected=json.loads(file.read_text());ds=file.parents[3].name;h=int(file.parents[2].name[1:]);seed=int(file.parents[1].name[4:]);base=file.parent.name
        complete=json.loads((file.parent/'completed.json').read_text());assert complete['legacy_operator_metric_replay_passed']
        for method in ['query_expanded','nlms_expanded','history_clip','cosa_expanded']:
            config=selected[method]['config']
            rows.append({'dataset':ds,'horizon':h,'seed':seed,'base':base,'method':method,'disabled':config.get('disabled',False),'alpha':config.get('alpha'),'rate':config.get('rate'),'steps':config.get('steps'),'validation_mae':selected[method]['mae'],'cosa_active_candidates':complete['cosa_active_candidates']})
    frame=pd.DataFrame(rows);frame.to_csv(A/'selected_configurations.csv',index=False)
    counts={}
    for name,sub in frame.groupby('method'):
        counts[name]={'disabled':int(sub.disabled.sum()),'at_alpha_1':int((sub.alpha==1).sum()),'at_rate_0_01':int((sub.rate==.01).sum()),'at_steps_50':int((sub.steps==50).sum()),'at_steps_20':int((sub.steps==20).sum())}
    counts['cosa_candidates_min']=int(frame.cosa_active_candidates.min());counts['cosa_candidates_max']=int(frame.cosa_active_candidates.max())
    return counts

if __name__=='__main__':
    result={'selection':selections(),'phases':{phase:audit_phase(phase) for phase in ['test','independent','withheld']}}
    manifest=json.loads((HERE/'withheld_data_manifest.json').read_text());lock=json.loads((HERE/'selection_lock.json').read_text())
    assert manifest['selection_lock_sha256']==sha(HERE/'selection_lock.json')
    assert manifest['prepared_utc']>lock['locked_utc']
    result['withheld_manifest']=manifest;result['selection_lock_utc']=lock['locked_utc']
    result['mase_computational_amendment']=json.loads((HERE/'mase_computational_amendment.json').read_text())
    result['unused_entity_variation']=json.loads((HERE/'unused_entity_variation_audit.json').read_text())
    assert sha(HERE/'locked_implementation/run_strengthening_validation.py')==lock['runner_sha256']
    result['locked_validation_implementation_sha256_matches']=True
    rows=[]
    for phase,label in [('test','Original benchmark'),('independent','EIA re-evaluation'),('withheld','Unused entities')]:
        m=pd.read_csv(A/f'{phase}_macro_comparisons.csv').query('block_days==7')
        for baseline in NAMES:
            sub=m[(m.baseline==baseline)&(m.metric=='mae')].iloc[0]
            rows.append([label,NAMES[baseline],interval(sub),len(CELLS) if phase=='test' else 4 if phase=='independent' else 8])
    (T/'expanded_macro_mae.tex').write_text(tabular(['Evaluation','Comparator','Mean MAE difference (\\%) [95\\% CI]','Settings'],rows,'llrl'),encoding='utf-8')
    (A/'evidence_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['selection'],indent=2));print('ALL SCIENTIFIC EVIDENCE AUDITS PASSED')
