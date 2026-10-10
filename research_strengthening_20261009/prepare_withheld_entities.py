from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,zipfile
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
P=json.loads((OUT/'protocol.json').read_text())
assert (OUT/'selection_lock.json').exists(), 'Selections must be locked before opening withheld outcomes'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
membership=json.loads((OUT/'existing_entity_membership.json').read_text())
for dataset,ids in P['withheld_entities'].items():
    for name,columns in membership.items():
        if dataset in name:
            used=[int(re.findall(r'\d+',c)[-1]) for c in columns if re.findall(r'\d+',c)]
            assert not set(ids)&set(used),(name,ids,used)

def hourly(frame,frequency,aggregation):
    assert not frame.index.has_duplicates
    groups=frame.resample('h',closed='right',label='right')
    count=groups.count();value=groups.mean() if aggregation=='mean' else groups.sum(min_count=frequency)
    return value.where(count==frequency)

folder=OUT/'data';folder.mkdir(exist_ok=True)
manifest={'prepared_utc':datetime.now(timezone.utc).isoformat(),'selection_lock_sha256':sha(OUT/'selection_lock.json'),'datasets':{},'entity_membership_overlap':False,'evaluation':'predeclared unused-entity cohort on existing historical calendar; fixed positional transfer; no source retraining'}
for ds,ids in P['withheld_entities'].items():
    source=ROOT/'data'/f'raw_{ds}.zip'
    if ds=='ecl':
        columns=[f'MT_{i:03}' for i in ids]
        with zipfile.ZipFile(source) as z:
            member=next(n for n in z.namelist() if n.lower().endswith('ld2011_2014.txt'))
            header=pd.read_csv(z.open(member),sep=';',decimal=',',nrows=0)
            timecol=header.columns[0]
            assert set(columns)<=set(header.columns)
            raw=pd.read_csv(z.open(member),sep=';',decimal=',',usecols=[timecol,*columns])
        frame=raw[columns].apply(pd.to_numeric,errors='coerce');frame.index=pd.DatetimeIndex(pd.to_datetime(raw[timecol]))
        frame=hourly(frame,4,'mean');units='kW'
    else:
        tables=[]
        with zipfile.ZipFile(source) as z:
            members=sorted(n for n in z.namelist() if re.search(r'Solar home 20\d\d-20\d\d\.csv$',n))
            for member in members:
                for chunk in pd.read_csv(z.open(member),skiprows=1,chunksize=20000):
                    raw=chunk[chunk['Consumption Category'].astype(str).str.upper().eq('GC')&chunk.Customer.isin(ids)]
                    if raw.empty:continue
                    clocks=[c for c in raw if re.fullmatch(r'\d{1,2}:\d{2}',str(c))]
                    long=raw.melt(id_vars=['Customer','date'],value_vars=clocks,var_name='clock',value_name='load')
                    end=long.clock.where(long.clock!='0:00','24:00')
                    long['timestamp']=pd.to_datetime(long.date,dayfirst=True,format='mixed')+pd.to_timedelta(end+':00')
                    tables.append(long[['timestamp','Customer','load']])
        long=pd.concat(tables)
        assert not long.duplicated(['timestamp','Customer']).any()
        frame=long.pivot(index='timestamp',columns='Customer',values='load').sort_index().reindex(columns=ids)
        frame.columns=[f'customer_{i:03}' for i in ids]
        frame=hourly(frame,2,'sum');units='kWh per hour-long interval'
    frame=frame.reindex(pd.date_range(frame.index.min(),frame.index.max(),freq='h'))
    values=frame.to_numpy(np.float32)
    file=folder/f'{ds}_entities101_116.npz'
    np.savez_compressed(file,values=values,observed=np.isfinite(values),timestamps=frame.index.to_numpy(),columns=np.array(frame.columns,str))
    manifest['datasets'][ds]={'entities':ids,'columns':frame.columns.tolist(),'shape':list(values.shape),'units':units,'first_timestamp':str(frame.index[0]),'last_timestamp':str(frame.index[-1]),'missing_by_channel':(~np.isfinite(values)).sum(0).tolist(),'source_archive_sha256':sha(source),'prepared_sha256':sha(file),'selection_did_not_use_test_completeness':True}
(OUT/'withheld_data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
