"""Lock all selected configurations, then retrieve the independent EIA period."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,urllib.request
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/functional_calibration_20260924'
selected=sorted((R/'select').glob('*/h*/seed*/*/selection.json'))
assert len(selected)==60,f'Expected all 60 frozen selections, found {len(selected)}'
lock={'locked_utc':datetime.now(timezone.utc).isoformat(),
      'protocol_sha256':hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest(),
      'files':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in selected],
      'independent_first_target':'2026-07-01 01:00:00','independent_last_target':'2026-09-01 00:00:00'}
lp=R/'selection_lock.json'
if lp.exists():
    old=json.loads(lp.read_text())
    assert old['files']==lock['files'] and old['protocol_sha256']==lock['protocol_sha256']
else:lp.write_text(json.dumps(lock,indent=2),encoding='utf-8')

url='https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2026_Jul_Dec.csv'
target=ROOT/'data/eia930/EIA930_BALANCE_2026_Jul_Dec.csv'
if not target.exists():
    part=target.with_suffix('.download')
    req=urllib.request.Request(url,headers={'User-Agent':'stable-shift-research/1.0'})
    with urllib.request.urlopen(req,timeout=45) as response,part.open('wb') as out:
        headers=dict(response.headers)
        while chunk:=response.read(1024*1024):out.write(chunk)
    part.replace(target)
else:headers={}

with np.load(ROOT/'data/prepared_v2/eia930_2023_2026.npz',allow_pickle=False) as z:
    old={k:z[k].copy() for k in z.files}
regions=old['columns'].astype(str).tolist()
frame=pd.read_csv(target,usecols=['Balancing Authority','UTC Time at End of Hour','Demand (MW)'],low_memory=False)
frame=frame[frame['Balancing Authority'].isin(regions)].copy()
frame['timestamp']=pd.to_datetime(frame['UTC Time at End of Hour'],utc=True).dt.tz_localize(None)
frame['demand']=pd.to_numeric(frame['Demand (MW)'].astype(str).str.replace(',','',regex=False),errors='coerce')
pivot=frame.pivot_table(index='timestamp',columns='Balancing Authority',values='demand',aggfunc='mean').reindex(columns=regions)
timeline=pd.date_range('2026-07-01 01:00','2026-09-01 00:00',freq='h')
new=pivot.reindex(timeline)
observed=new.notna()&(new>=0)
values=new.where(observed).to_numpy(np.float32)
assert pd.Timestamp(old['timestamps'][-1])+pd.Timedelta(hours=1)==timeline[0]
assert observed.to_numpy().sum()>0.8*values.size,'Independent period has inadequate provider coverage'
dest=ROOT/'data/prepared_v4/eia930_through_august_2026.npz';dest.parent.mkdir(exist_ok=True)
np.savez_compressed(dest,values=np.concatenate([old['values'],values]),
 timestamps=np.concatenate([old['timestamps'],timeline.to_numpy().astype('datetime64[ns]')]),
 columns=old['columns'],observed=np.concatenate([old['observed'],observed.to_numpy(bool)]))
manifest={'source_url':url,'downloaded_utc':datetime.now(timezone.utc).isoformat(),
 'headers':headers,'download_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
 'prepared_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'new_shape':list(values.shape),
 'observed_fraction':float(observed.to_numpy().mean()),'regions':regions,
 'first_new_timestamp':str(timeline[0]),'last_new_timestamp':str(timeline[-1]),
 'original_archive_preserved':True,'selection_lock_sha256':hashlib.sha256(lp.read_bytes()).hexdigest()}
(R/'independent_data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k!='headers'},indent=2))
