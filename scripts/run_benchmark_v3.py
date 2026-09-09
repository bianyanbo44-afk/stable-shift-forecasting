"""Resumable benchmark with source-pinned checkpoints and streaming evaluation."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from stable_shift_forecasting.benchmark_data import load_benchmark_data
from stable_shift_forecasting.config import seed_everything
from stable_shift_forecasting.model import StableShiftForecaster
from stable_shift_forecasting.upstream_baselines import build_upstream, VENDOR
from stable_shift_forecasting.training import _masked_load_view, _masked_mse, drift_score
from stable_shift_forecasting.scenarios import apply_scenario
from stable_shift_forecasting.deployment import OnlineSession


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def source_digest():
    files = [Path(__file__), *sorted((ROOT/'src/stable_shift_forecasting').glob('*.py'))]
    for folder in ['models', 'layers', 'utils']:
        files.extend(sorted((VENDOR/folder).glob('*.py')))
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_model(name, data, horizon, config):
    if name.startswith('StableShift'):
        return StableShiftForecaster(data.features.shape[-1], data.targets.shape[-1], horizon,
                                     context_length=config['context_length'], **config['stable_model'])
    return build_upstream(name, data.targets.shape[-1], horizon, config['context_length'], config['upstream_width'])


def forward(model, context, future):
    if isinstance(model, StableShiftForecaster):
        return model(context)
    return model(context, future)


def train_trial(name, data, train_set, val_set, horizon, seed, lr, config, path, signature):
    done = path/'completed.json'
    if done.exists():
        record = json.loads(done.read_text())
        if record['signature'] != signature:
            raise RuntimeError(f'Source/config mismatch at {done}; use a new output directory')
        return record
    path.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = make_model(name, data, horizon, config).to(config['device'])
    initial_hash = state_digest(model)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, generator=generator)
    # TimesNet selects FFT periods across a batch. Match deployment's one
    # origin at a time when selecting this model; other models are invariant.
    val_loader = DataLoader(val_set, batch_size=1 if name=='TimesNet' else config['batch_size'], shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config['weight_decay'])
    best = float('inf'); best_epoch = 0; history = []; start_time = time.perf_counter()
    for epoch in range(1, config['epochs']+1):
        model.train(); loss_sum = 0.; count = 0
        for batch in train_loader:
            x = batch['context'].to(config['device']); y = batch['target'].to(config['device'])
            future = batch['future_calendar'].to(config['device'])
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, StableShiftForecaster):
                output = model(x, return_representations=True)
                loss = (output.prediction-y).square().mean()
                if name != 'StableShiftNoStability':
                    augmented = x.clone()
                    augmented[..., :model.target_dim] += .01*torch.randn_like(x[..., :model.target_dim])
                    _, a, _ = model.encode(x); _, b, _ = model.encode(augmented)
                    loss = loss + config['consistency_weight']*(a-b).square().mean()
                    loss = loss + config['contraction_weight']*output.stable_representation.square().mean()
                masked, mask = _masked_load_view(model, x, observed=batch['context_observed'])
                loss = loss + config['reconstruction_weight']*_masked_mse(model.reconstruct(masked), x[..., :model.target_dim], mask)
            else:
                loss = (forward(model, x, future)-y).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss {path} epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            loss_sum += float(loss.detach())*len(x); count += len(x)
        model.eval(); val_total = 0.; val_count = 0
        # Model selection uses mean absolute error in original units, common
        # across methods. No test values are accessed during selection.
        with torch.no_grad():
            for batch in val_loader:
                x = batch['context'].to(config['device']); y = batch['target'].to(config['device'])
                output = forward(model, x, batch['future_calendar'].to(config['device']))
                error = (output-y).abs()*torch.as_tensor(data.std, device=config['device'])
                val_total += float(error.sum()); val_count += error.numel()
        val_mae = val_total/val_count
        if not np.isfinite(val_mae): raise FloatingPointError(f'Invalid validation loss {path}')
        history.append({'epoch': epoch, 'train_objective': loss_sum/count, 'validation_mae': val_mae,
                        'elapsed_seconds': time.perf_counter()-start_time})
        if val_mae < best:
            best = val_mae; best_epoch = epoch
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            checkpoint = {'state': state, 'signature': signature, 'method': name, 'seed': seed,
                          'horizon': horizon, 'lr': lr, 'epoch': epoch, 'validation_mae': best,
                          'initial_hash': initial_hash, 'config': config, 'data_manifest': data.manifest}
            torch.save(checkpoint, path/'best.tmp.pt'); (path/'best.tmp.pt').replace(path/'best.pt')
        pd.DataFrame(history).to_csv(path/'training.csv', index=False)
        print(json.dumps({'method': name, 'seed': seed, 'horizon':horizon, 'lr':lr,
                          'epoch':epoch, 'validation_mae':val_mae, 'best_epoch':best_epoch}), flush=True)
        if epoch-best_epoch >= config['patience']: break
    record = {'signature': signature, 'method':name, 'lr':lr, 'initial_hash':initial_hash,
              'best_validation_mae':best, 'best_epoch':best_epoch, 'epochs_run':epoch,
              'hit_epoch_budget':epoch==config['epochs'], 'checkpoint':str((path/'best.pt').relative_to(ROOT)),
              'training_seconds':time.perf_counter()-start_time, 'parameters':sum(p.numel() for p in model.parameters())}
    write_json(done, record)
    del optimizer, model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return record


def drift_reference(data, train_set, validation_set, config):
    # data are standardized from observed training values only.
    reference_mean = torch.zeros(data.targets.shape[-1], device=config['device'])
    reference_std = torch.ones_like(reference_mean)
    values = []
    for batch in DataLoader(validation_set, batch_size=config['batch_size']):
        values.extend(drift_score(batch['context'].to(config['device']),reference_mean,reference_std,len(reference_mean)).cpu().numpy())
    threshold = float(np.quantile(values, .95))
    changes = [float(np.abs(np.diff(train_set[i]['context'].numpy()[..., :len(reference_mean)], axis=0)).mean()) for i in range(len(train_set))]
    return reference_mean, reference_std, threshold, float(np.quantile(changes,.95))


def evaluate_stream(name, model, dataset, data, scenario, config, reference, variant):
    refmean, refstd, threshold, shock_threshold = reference
    session = None
    if isinstance(model, StableShiftForecaster):
        session = OnlineSession(model,refmean,refstd,threshold,config['adapter_learning_rate'],
                                update=variant=='online',anchor_only=variant=='anchor_only')
    rows = []; predictions = []; device = config['device']
    if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    for batch in DataLoader(dataset,batch_size=1,shuffle=False):
        x = batch['context'].to(device); origin = int(batch['origin'][0])
        # Scenario only transforms observed inputs. Deployment receives no scenario name.
        transformed = apply_scenario(x, data.targets.shape[-1], scenario, seed=2026+origin).context
        if device.startswith('cuda'): torch.cuda.synchronize()
        start = time.perf_counter()
        telemetry = {'drift_score':np.nan,'updated':False,'reconstruction_before':np.nan,'reconstruction_after':np.nan}
        if session is not None:
            prediction, telemetry = session.forecast(transformed, origin, observed=batch['context_observed'])
        elif model is None:
            period = 168 if name=='WeeklyNaive' else 24
            prediction = transformed[:, -period:, :data.targets.shape[-1]].repeat(1, (dataset.horizon+period-1)//period,1)[:, :dataset.horizon]
        else:
            model.eval()
            with torch.no_grad(): prediction = forward(model,transformed,batch['future_calendar'].to(device))
        if device.startswith('cuda'): torch.cuda.synchronize()
        elapsed = 1000*(time.perf_counter()-start)
        prediction = prediction.detach().cpu().numpy().astype(np.float64)*data.std+data.mean
        # Future labels first enter the calculation here, after prediction.
        target = batch['target'].numpy().astype(np.float64)*data.std+data.mean
        error = prediction-target
        if not np.isfinite(error).all(): raise FloatingPointError('Nonfinite test prediction')
        # Always deploy at every scheduled origin. Availability of future labels
        # changes scoring only and never changes the adapter's update history.
        scored = bool(batch['target_observed'].all())
        base = {'origin':origin, 'timestamp':str(data.timestamps[origin]),
                'scored':scored,
                'mae':float(np.abs(error).mean()) if scored else np.nan,
                'mse':float(np.square(error).mean()) if scored else np.nan,
                'rmse':float(np.sqrt(np.square(error).mean())) if scored else np.nan,
                'mase':float((np.abs(error)/data.mase_scale).mean()) if scored else np.nan,
                'natural_shock':float(x[..., :data.targets.shape[-1]].diff(dim=1).abs().mean())>=shock_threshold,
                'latency_ms':elapsed, **telemetry}
        rows.append(base); predictions.append(prediction[0].astype(np.float32))
    return pd.DataFrame(rows), np.stack(predictions), (torch.cuda.max_memory_allocated()/1024**2 if device.startswith('cuda') else None)


def run(config, datasets, horizons, seeds, methods):
    torch.set_num_threads(config['num_threads'])
    if config['device'].startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    out = ROOT/config['output']; out.mkdir(parents=True,exist_ok=True)
    digest = source_digest()
    signature = hashlib.sha256((digest+json.dumps(config,sort_keys=True)).encode()).hexdigest()
    protocol = out/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())['signature'] != signature:
        raise RuntimeError('Protocol/source changed; select a fresh output directory')
    write_json(protocol, {'signature':signature,'source_digest':digest,'config':config,
                         'python':platform.python_version(),'torch':torch.__version__,
                         'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None})
    # Preserve exact training/evaluation sources once, even if later work adds
    # manuscript-only tooling while these experiments are running.
    import shutil
    for source in [Path(__file__), *sorted((ROOT/'src/stable_shift_forecasting').glob('*.py'))]:
        archived=out/'source_snapshot'/source.relative_to(ROOT)
        if not archived.exists():
            archived.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,archived)
    for dataset_name in datasets:
        definition = config['datasets'][dataset_name]
        data = load_benchmark_data(ROOT/definition['path'],dataset_name,definition.get('target_limit'))
        manifest_path=out/dataset_name/'data_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text())['sha256']!=data.manifest['sha256']:
            raise RuntimeError('Dataset changed since the first run')
        write_json(manifest_path,data.manifest)
        for horizon in horizons:
            L = config['context_length']
            train = data.windows('train',L,horizon,config['train_stride'])
            val = data.windows('validation',L,horizon,config['validation_stride'])
            test = data.windows('test',L,horizon,horizon,include_unobserved=True)
            detail = out/dataset_name/f'h{horizon}'
            write_json(detail/'origins.json', {'train':train.origins.tolist(),'validation':val.origins.tolist(),'test':test.origins.tolist()})
            reference = drift_reference(data,train,val,config)
            write_json(detail/'thresholds.json',{'drift_threshold':reference[2],'shock_threshold':reference[3]})
            for seed in seeds:
                for name in methods:
                    method_dir = detail/f'seed{seed}'/name
                    finished = method_dir/'completed.json'
                    if finished.exists():
                        if json.loads(finished.read_text())['signature'] != signature: raise RuntimeError('Signature mismatch')
                        continue
                    method_dir.mkdir(parents=True,exist_ok=True)
                    model = None; selection = None; initial = None
                    if name in ['SeasonalNaive','WeeklyNaive']:
                        period=168 if name=='WeeklyNaive' else 24
                        errors=[]
                        for batch in DataLoader(val,batch_size=config['batch_size']):
                            base=batch['context'][:,-period:,:data.targets.shape[-1]]
                            pred=base.repeat(1,(horizon+period-1)//period,1)[:,:horizon]
                            errors.append(((pred-batch['target']).abs().numpy()*data.std).reshape(-1))
                        selection={'method':name,'best_validation_mae':float(np.concatenate(errors).mean()),
                                   'parameters':0,'training_seconds':0.,'signature':signature}
                        write_json(method_dir/'selected.json',selection)
                    else:
                        records = [train_trial(name,data,train,val,horizon,seed,lr,config,method_dir/f'lr{lr:g}',signature) for lr in config['learning_rates']]
                        selection = min(records,key=lambda r:r['best_validation_mae'])
                        model = make_model(name,data,horizon,config).to(config['device'])
                        checkpoint = torch.load(ROOT/selection['checkpoint'],map_location=config['device'],weights_only=False)
                        model.load_state_dict(checkpoint['state']); model.eval()
                        initial = copy.deepcopy(model.state_dict())
                        write_json(method_dir/'selected.json',selection)
                    variants = ['online','no_update','no_filter','anchor_only'] if name=='StableShift' else (['online'] if name.startswith('StableShift') else ['frozen'])
                    summaries = []
                    for variant in variants:
                        for scenario in config['scenarios']:
                            if initial is not None:
                                model.load_state_dict(initial)
                                if isinstance(model,StableShiftForecaster):
                                    model.robust_filter = config['stable_model']['robust_filter']
                                    model.robust_clip = config['stable_model']['robust_clip']
                                    if variant=='no_filter': model.robust_filter=False; model.robust_clip=0.
                            # no_filter uses the same gated online rule as the full model.
                            deployment_variant = 'online' if variant=='no_filter' else variant
                            frame,predictions,peak = evaluate_stream(name,model,test,data,scenario,config,reference,deployment_variant)
                            frame['dataset']=dataset_name;frame['horizon']=horizon;frame['seed']=seed;frame['method']=name;frame['variant']=variant;frame['scenario']=scenario
                            destination=method_dir/variant/scenario;destination.mkdir(parents=True,exist_ok=True)
                            frame.to_csv(destination/'windows.csv',index=False)
                            np.savez_compressed(destination/'predictions.npz',prediction=predictions,origins=test.origins)
                            for subset, selected in [('all',frame),('natural_shock',frame[frame.natural_shock])]:
                                summaries.append({'variant':variant,'scenario':scenario,'subset':subset,'n_windows':int(selected.scored.sum()),
                                     'deployment_windows':len(selected),
                                     'mae':float(selected.mae.mean()) if selected.scored.any() else None,
                                     'rmse':float(np.sqrt(selected.mse.mean())) if selected.scored.any() else None,
                                     'mase':float(selected.mase.mean()) if selected.scored.any() else None,
                                     'updates':int(selected.updated.sum()),'peak_gpu_mb':peak,
                                     'latency_median_ms':float(selected.latency_ms.median()) if len(selected) else None})
                            print(json.dumps({'evaluated':name,'dataset':dataset_name,'horizon':horizon,'seed':seed,'variant':variant,'scenario':scenario,'mae':frame.mae.mean(),'updates':int(frame.updated.sum())}),flush=True)
                    write_json(finished,{'signature':signature,'selection':selection,'summaries':summaries})
                    del model,initial
                    if torch.cuda.is_available():torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/benchmark_v3.json')
    parser.add_argument('--datasets',nargs='+');parser.add_argument('--horizons',nargs='+',type=int)
    parser.add_argument('--seeds',nargs='+',type=int);parser.add_argument('--methods',nargs='+')
    parser.add_argument('--output');parser.add_argument('--epochs',type=int);parser.add_argument('--device')
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text())
    for key in ['output','epochs','device']:
        if getattr(args,key) is not None:config[key]=getattr(args,key)
    run(config,args.datasets or list(config['datasets']),args.horizons or config['horizons'],args.seeds or config['seeds'],args.methods or config['methods'])


if __name__=='__main__':main()
