"""Rebuild hourly benchmark arrays from raw public data with observed masks.

Intervals are labelled by their ending hour. ECL power is averaged across four
quarter-hour readings; Ausgrid interval energy is summed across two half-hours.
Incomplete hourly labels remain missing. Input forward fill occurs only later
inside the benchmark loader. No households are removed using future coverage.
"""
from pathlib import Path
import hashlib
import json
import zipfile
import re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def complete_hour(frame,frequency,aggregation):
    if frame.index.has_duplicates:
        raise ValueError('Duplicate source timestamps require explicit resolution')
    grouped=frame.resample('h',closed='right',label='right')
    count=grouped.count()
    value=grouped.mean() if aggregation=='mean' else grouped.sum(min_count=frequency)
    return value.where(count==frequency)


def save(name,frame,source,units,policy):
    destination=ROOT/'data/prepared_v3';destination.mkdir(parents=True,exist_ok=True)
    # Keep the full time axis and column identities, even when labels are absent.
    frame=frame.reindex(pd.date_range(frame.index.min(),frame.index.max(),freq='h'))
    values=frame.to_numpy(np.float32)
    path=destination/f'{name}.npz'
    np.savez_compressed(path,values=values,observed=np.isfinite(values),
                        timestamps=frame.index.to_numpy(),columns=np.asarray(frame.columns,str))
    record={'dataset':name,'source':str(source.relative_to(ROOT)),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'prepared_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'shape':list(values.shape),'columns':list(frame.columns),'units':units,
        'first':str(frame.index.min()),'last':str(frame.index.max()),
        'missing_per_column':(~np.isfinite(values)).sum(0).tolist(),'aggregation':policy}
    path.with_suffix('.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)


def main():
    source=ROOT/'data/raw_ecl.zip'
    with zipfile.ZipFile(source) as z:
        member=next(n for n in z.namelist() if n.lower().endswith('ld2011_2014.txt'))
        raw=pd.read_csv(z.open(member),sep=';',decimal=',',usecols=range(17))
    ecl=raw.iloc[:,1:].apply(pd.to_numeric,errors='coerce')
    ecl.index=pd.DatetimeIndex(pd.to_datetime(raw.iloc[:,0]))
    save('ecl',complete_hour(ecl,4,'mean'),source,'kW',
         'First 16 source channels; mean of four interval-ending power readings; complete hours only')

    source=ROOT/'data/raw_pjm_kaggle.zip'
    with zipfile.ZipFile(source) as z:
        raw=pd.read_csv(z.open('pjm_hourly_est.csv'),parse_dates=['Datetime'])
    pjm=raw.set_index('Datetime')[['PJME','PJMW']].sort_index()
    duplicates=pjm.index[pjm.index.duplicated(False)].unique()
    pjm=pjm[~pjm.index.duplicated(keep='first')]
    pjm.loc[duplicates,:]=np.nan
    # The two regions' shared collection period is determined by first and last
    # observations; holes within that period remain explicit missing labels.
    begin=max(pjm[c].first_valid_index() for c in pjm)
    end=min(pjm[c].last_valid_index() for c in pjm)
    save('pjm',pjm.loc[begin:end],source,'MW',
         'PJME/PJMW shared collection period; source clock labels; ambiguous repeated hours missing; no imputed targets')

    source=ROOT/'data/raw_ausgrid.zip';tables=[]
    with zipfile.ZipFile(source) as z:
        members=sorted(n for n in z.namelist() if re.search(r'Solar home 20\d\d-20\d\d\.csv$',n))
        for member in members:
            raw=pd.read_csv(z.open(member),skiprows=1)
            raw=raw[raw['Consumption Category'].astype(str).str.upper().eq('GC') & raw.Customer.isin(range(1,17))]
            clocks=[c for c in raw if re.fullmatch(r'\d{1,2}:\d{2}',str(c))]
            long=raw.melt(id_vars=['Customer','date'],value_vars=clocks,var_name='clock',value_name='load')
            # The final 0:00 column closes the stated day, following 23:30.
            interval_end=long.clock.where(long.clock!='0:00','24:00')
            long['timestamp']=pd.to_datetime(long.date,dayfirst=True,format='mixed')+pd.to_timedelta(interval_end+':00')
            tables.append(long[['timestamp','Customer','load']])
    long=pd.concat(tables)
    if long.duplicated(['timestamp','Customer']).any():raise ValueError('Duplicate household readings')
    aus=long.pivot(index='timestamp',columns='Customer',values='load').sort_index()
    aus.columns=[f'customer_{int(c):03d}' for c in aus]
    save('ausgrid',complete_hour(aus,2,'sum'),source,'kWh per hour',
         'Customers 1 through 16, GC; sum of two interval-ending energy readings; complete hours only')


if __name__=='__main__':main()
