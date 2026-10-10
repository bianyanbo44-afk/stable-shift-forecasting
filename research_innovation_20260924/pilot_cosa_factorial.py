"""Factorial separation of predeployment calibration and online parameter updates.

The reviewed COSA output module is used under a controlled daily, matured-label,
immutable-issuance protocol. This is not its native retrospective-adjustment
benchmark. All four arms have identical base forecasts and observable context.
"""
from pathlib import Path
import ast,copy,hashlib,json,sys,time,platform
from collections import deque
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stable_shift_forecasting.benchmark_data import load_benchmark_data
from stable_shift_forecasting.model import StableShiftForecaster
from stable_shift_forecasting.upstream_baselines import build_upstream

OUT=Path(__file__).resolve().parent/'cosa_factorial_pilot';OUT.mkdir(exist_ok=True)
UP=ROOT/'literature/kbs_extension_20260924/cosa/tta/cosa.py'
node=next(n for n in ast.parse(UP.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='SimpleOutputAdapter')
space={'torch':torch,'nn':nn,'ADAPTER_TYPES':('Linear','MLP')}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(UP),'exec'),space)
COSA=space['SimpleOutputAdapter']
CELLS=[('eia930',24,'results/benchmark_v3_formal','data/prepared_v2/eia930_2023_2026.npz',7),
('eia930',168,'results/benchmark_v3_formal','data/prepared_v2/eia930_2023_2026.npz',7),
('ecl',24,'cloud_returned_20260909/results/cloud_ecl_v3','data/prepared_v3/ecl.npz',16),
('ecl',168,'cloud_returned_20260909/results/cloud_ecl_h168_v3','data/prepared_v3/ecl.npz',16),
('ausgrid',24,'cloud_returned_20260909/results/cloud_ausgrid_v3','data/prepared_v3/ausgrid.npz',16),
('ausgrid',168,'cloud_returned_20260909/results/cloud_ausgrid_v3','data/prepared_v3/ausgrid.npz',16)]
CONFIGS=[(.0003,1),(.0003,5),(.001,1),(.001,5)]
BASES=['StableShift','DLinear'];DEVICE='cuda';SEED=11;CAL_EPOCHS=10
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.5)
START=time.monotonic()


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def budget():
    if time.monotonic()-START>900:raise TimeoutError('Fixed 15-minute factorial pilot limit')


def fast_forward(adapter,pred,ctx):
    # Exact vectorization of the upstream Linear, variable-wise branch.
    assert adapter.var_wise and adapter.adapter_type=='Linear'
    x=torch.cat([pred.transpose(1,2),ctx[:,None].expand(-1,pred.shape[-1],-1)],-1)
    weight=torch.stack([m.weight for m in adapter.fc_layers])
    bias=torch.stack([m.bias for m in adapter.fc_layers])
    residual=torch.einsum('bdi,dhi->bdh',x,weight)+bias[None]
    return pred+(residual*torch.tanh(adapter.gate)[None,:,None]).transpose(1,2)


class Updater:
    def __init__(self,model,rate,steps):
        self.model=model;self.rate=rate;self.steps=steps
        self.optimizer=torch.optim.Adam(model.parameters(),lr=rate,weight_decay=0.,foreach=True)
        self.history=deque(maxlen=5);self.current_lr=rate;self.count=0

    def update(self,pred,target,mask,ctx,steps=None):
        steps=self.steps if steps is None else steps
        if not bool(mask.any()):return
        for step in range(steps):
            output=fast_forward(self.model,pred,ctx)
            loss=((output-target).square()*mask).sum()/mask.sum()
            loss=loss+1e-4*sum(p.square().sum() for p in self.model.parameters())
            value=float(loss.detach())
            self.history.append(value)
            if step==0:self.current_lr=self.rate
            elif len(self.history)>=2:
                recent=list(self.history)[-3:]
                trend=recent[-1]-recent[0];variance=float(np.var(recent,ddof=1))
                if trend>0 and variance<1e-6:self.current_lr=min(self.current_lr*1.2,.005)
                elif trend<-.01:self.current_lr=min(self.current_lr*1.05,.005)
                elif abs(trend)<1e-6:self.current_lr=max(self.current_lr*.8,.0001)
                factor=.5*(1+np.cos(step*3.14159/self.steps))
                self.current_lr=.0001+(self.current_lr-.0001)*factor
            for group in self.optimizer.param_groups:group['lr']=self.current_lr
            self.optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),max(.05,min(.5,value)))
            self.optimizer.step();self.count+=1
            if step>2 and len(self.history)>1 and abs(self.history[-1]-self.history[-2])<1e-4:break


def prepare(cell,base):
    ds,h,root,datafile,d=cell
    data=load_benchmark_data(ROOT/datafile,ds,d)
    path=ROOT/root/ds/f'h{h}'/f'seed{SEED}'/base
    chosen=json.loads((path/'selected.json').read_text())
    checkpoint=path/f"lr{chosen['lr']:g}"/'best.pt'
    cfg=json.loads((ROOT/root/'protocol.json').read_text())['config']
    if base=='StableShift':model=StableShiftForecaster(data.features.shape[1],d,h,context_length=336,**cfg['stable_model'])
    else:model=build_upstream(base,d,h,336,cfg['upstream_width'])
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state'])
    model.to(DEVICE).eval()
    for p in model.parameters():p.requires_grad_(False)
    origins=np.arange(data.train_end-h,data.validation_end-h+1,24)
    predictions=[]
    with torch.no_grad():
        for i in range(0,len(origins),32):
            budget()
            x=torch.tensor(np.stack([data.features[o-336:o] for o in origins[i:i+32]]),device=DEVICE)
            p=model(x)
            predictions.append(p)
    pred=torch.cat(predictions).detach()
    targets=torch.tensor(np.stack([data.targets[o:o+h] for o in origins]),device=DEVICE)
    observed=torch.tensor(np.stack([data.observed[o:o+h] for o in origins]),device=DEVICE)
    ctx=[];hist=deque(maxlen=5);j=0;context_label_max=[]
    for o in origins:
        while j<len(origins) and origins[j]+h<=o:
            mask=observed[j]
            if bool(mask.any()):hist.append(float(targets[j][mask].mean()))
            j+=1
        values=list(hist)[::-1]
        values=values+[values[-1]]*(5-len(values)) if values else [0.]*5
        ctx.append(values)
        context_label_max.append(int(origins[j-1]+h-1) if j else -1)
    assert all(m<o for m,o in zip(context_label_max,origins))
    split=data.train_end+24*((data.validation_end-data.train_end)//48)
    train_ix=np.flatnonzero((origins>=data.train_end)&(origins+h<=split))
    eval_ix=np.flatnonzero(origins>=split)
    valid=observed.all((1,2))
    del model;torch.cuda.empty_cache()
    return data,origins,pred,targets,observed,torch.tensor(ctx,device=DEVICE),train_ix,eval_ix,valid,checkpoint


def run(cell,base):
    ds,h,root,datafile,d=cell
    data,origins,pred,truth,mask,ctx,tr,ev,scored,checkpoint=prepare(cell,base)
    digest=sha(checkpoint)
    dest=OUT/ds/f'h{h}'/base;dest.mkdir(parents=True,exist_ok=True)
    std=torch.tensor(data.std,device=DEVICE)
    metrics=[];windowrows=[];events=[]
    for rate,steps in CONFIGS:
        budget();torch.manual_seed(SEED)
        initial=COSA(h,buffer_context_size=5,n_vars=d,var_wise_gating=True,adapter_type='Linear').to(DEVICE)
        with torch.no_grad():
            torch.testing.assert_close(initial(pred[:3],ctx[:3]),fast_forward(initial,pred[:3],ctx[:3]),rtol=1e-6,atol=1e-6)
        warm=copy.deepcopy(initial);fit=Updater(warm,rate,steps)
        generator=torch.Generator().manual_seed(SEED)
        for epoch in range(CAL_EPOCHS):
            for order in torch.randperm(len(tr),generator=generator).split(32):
                ix=tr[order.numpy()]
                fit.update(pred[ix],truth[ix],mask[ix],ctx[ix],steps=1)
        with torch.no_grad():
            torch.testing.assert_close(warm(pred[ev[:3]],ctx[ev[:3]]),fast_forward(warm,pred[ev[:3]],ctx[ev[:3]]),rtol=2e-6,atol=2e-6)
        for pretrained,updates in [(False,False),(False,True),(True,False),(True,True)]:
            adapter=copy.deepcopy(warm if pretrained else initial)
            runner=Updater(adapter,rate,steps)
            if pretrained:
                runner.optimizer.load_state_dict(copy.deepcopy(fit.optimizer.state_dict()))
            name=('warm' if pretrained else 'cold')+('_online' if updates else '_frozen')
            last=-1;pending=0;issued=[];m=[];s=[]
            for i in ev:
                o=int(origins[i]);budget()
                if updates:
                    while pending<len(issued) and origins[issued[pending]]+h<=o:
                        j=issued[pending]
                        label_max=int(origins[j]+h-1)
                        assert label_max<o
                        runner.update(pred[j:j+1],truth[j:j+1],mask[j:j+1],ctx[j:j+1])
                        last=label_max
                        events.append({'dataset':ds,'horizon':h,'base':base,'rate':rate,'steps':steps,
                            'arm':name,'event_origin':o,'forecast_origin':int(origins[j]),'maximum_label_index':label_max})
                        pending+=1
                with torch.no_grad():
                    forecast=fast_forward(adapter,pred[i:i+1],ctx[i:i+1])[0]
                    error=(forecast-truth[i])*std
                    mae=float(error.abs().mean()) if bool(scored[i]) else np.nan
                    mse=float(error.square().mean()) if bool(scored[i]) else np.nan
                m.append(mae);s.append(mse);issued.append(int(i))
                windowrows.append({'dataset':ds,'horizon':h,'base':base,'rate':rate,'steps':steps,'arm':name,
                    'origin':o,'scored':bool(scored[i]),'mae':mae,'mse':mse,'max_update_label_index':last})
            metrics.append({'dataset':ds,'horizon':h,'base':base,'rate':rate,'steps':steps,'arm':name,
                'mae':float(np.nanmean(m)),'rmse':float(np.sqrt(np.nanmean(s))),
                'parameters':sum(p.numel() for p in adapter.parameters()),'calibration_steps':fit.count if pretrained else 0,
                'deployment_steps':runner.count})
        torch.save(warm.state_dict(),dest/f'warm_lr{rate:g}_steps{steps}.pt')
    assert digest==sha(checkpoint)
    pd.DataFrame(windowrows).to_csv(dest/'windows.csv',index=False)
    pd.DataFrame(events).to_csv(dest/'events.csv',index=False)
    pd.DataFrame(metrics).to_csv(dest/'metrics.csv',index=False)
    write(dest/'manifest.json',{'source_checkpoint':str(checkpoint.relative_to(ROOT)),'source_sha256':digest,
        'data_sha256':data.manifest['sha256'],'calibration_origins':origins[tr].tolist(),'evaluation_origins':origins[ev].tolist(),
        'maximum_context_label_index_precedes_origin':True,'all_update_labels_precede_issue':True})
    frame=pd.DataFrame(metrics)
    common=frame.groupby(['rate','steps']).mae.mean().sort_values().index[0]
    chosen=frame[(frame.rate==common[0])&(frame.steps==common[1])]
    write(dest/'common_selection.json',{'rate':float(common[0]),'steps':int(common[1]),'selection':'minimum mean MAE across the four arms, one common configuration'})
    print(chosen[['dataset','horizon','base','arm','mae','deployment_steps']].to_string(index=False),flush=True)
    return metrics


def main():
 write(OUT/'protocol.json',{'date':'2026-09-24','written_before_run':True,'source_sha256':sha(Path(__file__)),
 'hypothesis':'Separating initial calibration from deployment-time parameter updates identifies the marginal contribution of ongoing adaptation',
 'factorial_arms':['cold_frozen','cold_online','warm_frozen','warm_online'],
 'same_information':'Same frozen base predictions and contemporaneously available COSA context vector in every arm',
 'source':'COSA SimpleOutputAdapter, Linear/variable-wise, buffer 5; pinned commit 527c0feb9e997dd85af485ee027616b446e4ae77',
 'license':'Pinned repository LICENSE is CC BY-NC-SA 4.0; README differs; exact source license retained',
 'protocol_adaptation':'Daily issuance, updates only after a complete prior forecast horizon has elapsed, no retrospective replacement; common schedule disables PAAS/partial-label updates',
 'optimization':'Adam, original dynamic clipping, L2 1e-4, within-event CALR; fixed ten calibration epochs',
 'configs':CONFIGS,'bases':BASES,'cells':[[c[0],c[1]] for c in CELLS],'seeds':[SEED],
 'selection':'Original validation first half calibrates, second half scores; select one common configuration by mean four-arm MAE, keeping hyperparameters identical within each factorial comparison',
 'information_caveat':'Base selection used the original full validation period; results are development screening, not independent confirmation',
 'resource_limit':'900 seconds, four CPU threads, half GPU memory','torch':torch.__version__,'python':platform.python_version()})
 rows=[]
 for cell in CELLS:
    for base in BASES:rows.extend(run(cell,base))
 D=pd.DataFrame(rows);D.to_csv(OUT/'metrics.csv',index=False)
 write(OUT/'completed.json',{'seconds':time.monotonic()-START,'source_checkpoints_unchanged':True,'test_evaluated':False,'cells':12,'arms':4})
 print('COMPLETED',time.monotonic()-START,flush=True)

if __name__=='__main__':main()
