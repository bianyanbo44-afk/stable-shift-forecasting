"""Remove only the query bound at the already selected QNC step size."""
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_functional_study as s
from stable_shift_forecasting.benchmark_data import load_benchmark_data

START=time.monotonic()
data_cache={}
for phase in ['test','independent']:
    for cell in s.f.CELLS:
        ds,h,root,path,d=cell
        if phase=='independent' and ds!='eia930':continue
        if phase=='independent':path='data/prepared_v4/eia930_through_august_2026.npz'
        if (phase,ds) not in data_cache:data_cache[phase,ds]=load_benchmark_data(s.ROOT/path,ds,d)
        data=data_cache[phase,ds]
        for seed in s.SEEDS:
            for base in s.f.BASES:
                folder=s.OUT/phase/ds/f'h{h}'/f'seed{seed}'/base
                dest=folder/'nlms_matched'
                if (dest/'metrics.json').exists():continue
                with np.load(folder/'frozen/predictions.npz') as z:
                    origins=z['origins'];pred=torch.tensor(z['prediction'],device=s.f.DEVICE)
                truth=torch.tensor(np.stack([data.targets[o:o+h] for o in origins]),device=s.f.DEVICE)
                mask=torch.tensor(np.stack([data.observed[o:o+h] for o in origins]),device=s.f.DEVICE)
                ctx=[]
                for o in origins:
                    row=[]
                    for k in range(5):
                        end=int(o)-24*k;start=end-h;valid=data.observed[start:end]
                        row.append(float(data.targets[start:end][valid].mean()) if valid.any() else 0.)
                    ctx.append(row)
                ctx=torch.tensor(ctx,device=s.f.DEVICE)
                choice=json.loads((s.OUT/'select'/ds/f'h{h}'/f'seed{seed}'/base/'selection.json').read_text())['query']['config']
                config=dict(choice)
                if not config.get('disabled'):config['trust']=None
                result=s.evaluate(data,origins,pred,truth,mask,ctx,seed,'nlms',config,save=dest)
                s.f.write(dest/'ablation.json',{'source_query_config':choice,'modified_config':config,
                  'only_change':'query movement constraint removed; identical selected alpha and feedback schedule'})
        print('completed',phase,ds,h,flush=True)
print('DONE',time.monotonic()-START,flush=True)
