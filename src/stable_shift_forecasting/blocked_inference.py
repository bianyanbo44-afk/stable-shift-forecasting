"""Paired crossed seed/time-block bootstrap for saved forecast errors.

Missing origin slots stay on the regular forecast grid. Each resample uses
the same seed draws and time indices for both methods, and the same time
indices for all seed replicates. Seeds are not treated as independent data
windows. The percentile CI estimates uncertainty over seeds and time blocks.
"""
import numpy as np


def statistic(values, metric):
    value = np.nanmean(values, axis=(-2,-1))
    return np.sqrt(value) if metric=='rmse' else value


def paired_block_bootstrap(baseline, proposed, metric='mae', block_length=7,
                           resamples=5000, seed=2026):
    a, b = np.asarray(baseline,float), np.asarray(proposed,float)
    if a.ndim!=2 or a.shape!=b.shape:
        raise ValueError('Expected matched [seeds, regular origin slots] arrays')
    if not np.array_equal(np.isfinite(a),np.isfinite(b)):
        raise ValueError('Both methods must score identical labels')
    if np.isinf(a).any() or np.isinf(b).any() or (a<0).any() or (b<0).any():
        raise ValueError('Errors must be nonnegative or missing')
    if a.shape[1]<2 or not np.isfinite(a).any():
        raise ValueError('Insufficient paired observations')
    S,N=a.shape; block_length=min(max(1,block_length),N)
    blocks=int(np.ceil(N/block_length))
    rng=np.random.default_rng(seed)
    original_a=float(statistic(a,metric));original_b=float(statistic(b,metric))
    if original_a<=0:raise ValueError('Relative error requires positive baseline')
    diffs=[];ratios=[]
    # Retry only resamples containing no scored data, possible for rare shock subsets.
    attempts=0
    while len(diffs)<resamples:
        attempts+=1
        if attempts>resamples*20:raise ValueError('Too few observed time blocks')
        time_indices=((rng.integers(0,N,size=blocks)[:,None]+np.arange(block_length))%N).ravel()[:N]
        seeds=rng.integers(0,S,size=S)
        aa=a[seeds[:,None],time_indices[None,:]]
        bb=b[seeds[:,None],time_indices[None,:]]
        if not np.isfinite(aa).any():continue
        va,vb=float(statistic(aa,metric)),float(statistic(bb,metric))
        if va<=0:continue
        diffs.append(vb-va);ratios.append(vb/va-1)
    diffs=np.asarray(diffs);ratios=np.asarray(ratios)
    observed=original_b-original_a
    # Null-centred bootstrap approximation; add-one Monte Carlo correction.
    p=float((1+np.count_nonzero(np.abs(diffs-observed)>=abs(observed)))/(resamples+1))
    count=np.isfinite(b-a).sum(axis=0)
    per_origin=np.divide(np.nansum(b-a,axis=0), count,
                         out=np.full(N,np.nan), where=count>0)
    pairs=np.isfinite(per_origin[:-1])&np.isfinite(per_origin[1:])
    autocorrelation=None
    if pairs.sum()>2 and np.std(per_origin[:-1][pairs])>0 and np.std(per_origin[1:][pairs])>0:
        autocorrelation=float(np.corrcoef(per_origin[:-1][pairs],per_origin[1:][pairs])[0,1])
    return {'baseline_error':original_a,'proposed_error':original_b,'difference':observed,
            'ci_low':float(np.quantile(diffs,.025)),'ci_high':float(np.quantile(diffs,.975)),
            'relative_difference':original_b/original_a-1,
            'relative_ci_low':float(np.quantile(ratios,.025)),
            'relative_ci_high':float(np.quantile(ratios,.975)),
            'p_value':p,'block_length':block_length,'resamples':resamples,
            'seeds':S,'origin_slots':N,'observed_pairs':int(np.isfinite(a).sum()),
            'lag1_error_difference_correlation':autocorrelation}
