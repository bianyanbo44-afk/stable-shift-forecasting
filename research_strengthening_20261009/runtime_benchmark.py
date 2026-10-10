from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import torch
import run_strengthening as s

OUT=s.OUT;OLD=s.old
BASE=OLD.QueryCalibration
class HistoricalCalibration(BASE):
    def update(self,p,truth,mask,ctx,current,current_ctx):
        return super().update(p,truth,mask,ctx,p,ctx)

def measure(data,origins,pred,truth,mask,ctx,seed,method,config):
    OLD.QueryCalibration=HistoricalCalibration if method=='history_clip' else BASE
    try:
        kind='cosa' if method=='cosa_expanded' else ('frozen' if method=='frozen' else 'nlms')
        return OLD.evaluate(data,origins,pred,truth,mask,ctx,seed,kind,config)
    finally:OLD.QueryCalibration=BASE

rows=[]
destination=OUT/'analysis/runtime_repetitions.csv';destination.parent.mkdir(exist_ok=True)
if destination.exists():rows=pd.read_csv(destination).to_dict('records')
for number,cell in enumerate(OLD.f.CELLS):
    ds,h,*_=cell
    for base in s.P['sources']:
        # Avoid repeating an already complete synchronized benchmark.
        if sum(r['dataset']==ds and r['horizon']==h and r['base']==base for r in rows)==15:continue
        data,origins,pred,truth,mask,ctx=s.stream(cell,base,11,'test')
        selected=json.loads((OUT/'select'/ds/f'h{h}'/'seed11'/base/'selection.json').read_text())
        methods=['frozen','cosa_expanded','nlms_expanded','query_expanded','history_clip']
        configs={'frozen':{},**{m:selected[m]['config'] for m in methods if m!='frozen'}}
        for m in methods:measure(data,origins,pred,truth,mask,ctx,11,m,configs[m])
        for rep in range(3):
            shift=(number+rep)%len(methods);order=methods[shift:]+methods[:shift]
            for m in order:
                result=measure(data,origins,pred,truth,mask,ctx,11,m,configs[m])
                # This also checks selected batched operator results against a scalar recurrence.
                metrics=json.loads((OUT/'test'/ds/f'h{h}'/'seed11'/base/'frozen/metrics.json').read_text())
                if m=='cosa_expanded':metrics=json.loads((OUT/'test'/ds/f'h{h}'/'seed11'/base/'cosa_expanded/metrics.json').read_text())
                elif m not in ['frozen']:
                    mapping=json.loads((OUT/'test'/ds/f'h{h}'/'seed11'/base/'method_mapping.json').read_text())
                    if mapping[m]>=0:metrics=json.loads((OUT/'test'/ds/f'h{h}'/'seed11'/base/'operators.json').read_text())['metrics'][mapping[m]]
                for key in ['mae','rmse','mase']:assert np.isclose(result[key],metrics[key],rtol=2e-5,atol=1e-7),(ds,h,base,m,key,result[key],metrics[key])
                rows.append({'dataset':ds,'horizon':h,'base':base,'seed':11,'method':m,'repetition':rep,'seconds':result['seconds'],'origins':len(origins),'ms_per_origin':1000*result['seconds']/len(origins),'updates':result['updates'],'config':json.dumps(configs[m]),'excluded_source_inference':True})
        pd.DataFrame(rows).to_csv(destination,index=False)
        print('RUNTIME VERIFIED',ds,h,base,flush=True)
        if time.monotonic()-s.START>700:raise SystemExit(75)
summary=pd.DataFrame(rows).groupby(['dataset','horizon','base','method'])[['seconds','ms_per_origin']].median().reset_index()
summary.to_csv(OUT/'analysis/runtime_summary.csv',index=False)
print('RUNTIME COMPLETE',flush=True)
