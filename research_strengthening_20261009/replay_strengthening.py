"""Regenerate one selected study cell and compare with archived metrics."""
from pathlib import Path
import argparse,json
import numpy as np
import run_strengthening as s

parser=argparse.ArgumentParser()
parser.add_argument('--phase',choices=['test','independent','withheld'],default='withheld')
parser.add_argument('--dataset',choices=['eia930','ecl','ausgrid'],default='ausgrid')
parser.add_argument('--horizon',type=int,choices=[24,168],default=24)
parser.add_argument('--source',choices=['StableShift','DLinear'],default='StableShift')
parser.add_argument('--seed',type=int,choices=[11,22,33,44,55],default=11)
args=parser.parse_args()
s.verify_protected()
cell=next(c for c in s.old.f.CELLS if c[0]==args.dataset and c[1]==args.horizon)
selected=json.loads((s.OUT/'select'/args.dataset/f'h{args.horizon}'/f'seed{args.seed}'/args.source/'selection.json').read_text())
data,origins,pred,truth,mask,ctx=s.stream(cell,args.source,args.seed,args.phase)
saved=s.OUT/args.phase/args.dataset/f'h{args.horizon}'/f'seed{args.seed}'/args.source
dest=s.ROOT/'replays_strengthening'/args.phase/args.dataset/f'h{args.horizon}'/f'seed{args.seed}'/args.source
configs,mapping=s.evaluation_configs(selected)
actual=s.batch_operators(data,origins,pred,truth,mask,ctx,configs,dest/'operators.npz')
reference=json.loads((saved/'operators.json').read_text())['metrics']
for new,old in zip(actual,reference):
    assert new['config']==old['config']
    for metric in ['mae','rmse','mase']:assert np.isclose(new[metric],old[metric],rtol=2e-5,atol=1e-7),(new['config'],metric,new[metric],old[metric])
checks={}
for name,kind,config in [('frozen','frozen',{}),('cosa_expanded','cosa',selected['cosa_expanded']['config'])]:
    new=s.scalar_evaluate(data,origins,pred,truth,mask,ctx,args.seed,kind,config,save=dest/name)
    old=json.loads((saved/name/'metrics.json').read_text())
    for metric in ['mae','rmse','mase']:assert np.isclose(new[metric],old[metric],rtol=2e-5,atol=1e-7),(name,metric)
    checks[name]={key:new[key] for key in ['mae','rmse','mase']}
s.write(dest/'verification.json',{'arguments':vars(args),'operator_recurrences':len(actual),'all_metric_replays_passed':True,'scalar_checks':checks,'rtol':2e-5,'atol':1e-7})
print(json.dumps({'replay':vars(args),'operator_recurrences':len(actual),'passed':True},indent=2))
