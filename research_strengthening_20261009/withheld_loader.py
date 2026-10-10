"""Keep every declared entity; MASE is secondary on training-defined scales.

Chronology, training mean/std, past filling and features match the original
loader. Zero training seasonal scales do not exclude a channel from forecasts,
updates, MAE or RMSE. MASE alone has a separate, recorded observed count.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from stable_shift_forecasting.benchmark_data import BenchmarkData
from stable_shift_forecasting.data import _calendar_features

def load_withheld_data(path:Path,name:str,target_limit:int):
    with np.load(path,allow_pickle=False) as z:
        raw=z['values'][:,:target_limit].astype(np.float32);stamps=pd.DatetimeIndex(z['timestamps']);columns=z['columns'][:target_limit].astype(str)
        observed=z['observed'][:,:target_limit].astype(bool)
    assert stamps.is_monotonic_increasing and not stamps.has_duplicates
    timeline=pd.date_range(stamps[0],stamps[-1],freq='h')
    frame=pd.DataFrame(np.where(observed&np.isfinite(raw),raw,np.nan),index=stamps).reindex(timeline)
    raw=frame.to_numpy(np.float32);observed=np.isfinite(raw)
    train_end=int(.7*len(raw));validation_end=train_end+int(.1*len(raw))
    mean=np.nanmean(raw[:train_end],axis=0).astype(np.float32)
    raw_std=np.nanstd(raw[:train_end],axis=0).astype(np.float32)
    std=np.where(raw_std>1e-6,raw_std,1.).astype(np.float32)
    assert np.isfinite(mean).all()
    filled=frame.ffill().to_numpy(np.float32);warmup=int(np.flatnonzero(np.isfinite(filled).all(1))[0])
    filled[:warmup]=np.nan_to_num(filled[:warmup],nan=0.)
    calendar=_calendar_features(timeline);target=((filled-mean)/std).astype(np.float32)
    pairs=observed[24:train_end]&observed[:train_end-24]
    original_scale=np.nanmean(np.where(pairs,np.abs(raw[24:train_end]-raw[:train_end-24]),np.nan),axis=0).astype(np.float64)
    defined=np.isfinite(original_scale)&(original_scale>0);assert defined.any()
    # Infinity is an internal zero-contribution sentinel, not an invented scale.
    scales=np.where(defined,original_scale,np.inf)
    manifest={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dataset':name,'columns':columns.tolist(),'shape':list(raw.shape),
      'first_timestamp':str(timeline[0]),'last_timestamp':str(timeline[-1]),'train_end_exclusive':str(timeline[train_end]),'validation_end_exclusive':str(timeline[validation_end]),
      'leading_excluded_hours':warmup,'missing_by_channel':(~observed).sum(0).tolist(),'label_policy':'All observed entries for primary MAE/RMSE and feedback',
      'input_policy':'Past-only forward fill; leading incomplete contexts excluded','scale_policy':'Training-only moments; original unit-std fallback for constant training channels',
      'constant_training_channels':columns[raw_std<=1e-6].tolist(),'mase_defined_channels':columns[defined].tolist(),'mase_undefined_channels':columns[~defined].tolist(),
      'mase_scale_training':original_scale.tolist(),'mase_policy':'Secondary MASE uses only channels with positive finite training seasonal scale, separate observed_mase_count; primary entities and masks unchanged'}
    return BenchmarkData(np.c_[target,calendar].astype(np.float32),target,calendar,observed,timeline,mean,std,scales,train_end,validation_end,warmup,manifest)

def mase_counts(data,mask):
    defined=np.isfinite(data.mase_scale)&(data.mase_scale>0)
    return mask.detach().cpu().numpy()[:,:,defined].sum((1,2))
