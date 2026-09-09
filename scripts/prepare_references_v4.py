"""Select and verify the bibliography for the revised empirical manuscript."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, re, html
import requests, bibtexparser

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript_v4'
KEYS='kim2025tta chen2024calibration qiu2024tfb he2025robust hu2025adaptive murad2025wpmixer qiu2025duet fan2023dishts zeng2023transformers antoniadis2024hierarchical ratnam2015ausgrid ding2023peft wang2022continualtta forootani2026compact baydan2026operational liao2026survey das2023tide liu2023itransformer wu2023timesnet nie2023patchtst liu2022nonstationary miyato2018spectral hu2021lora uci2015ecl eia9302026 apt2026 adanodes2026 adamshift2026'.split()

def verify(e):
    doi=e.get('doi','')
    record={'key':e['ID'],'doi':doi,'checked_on':'2026-09-09'}
    if not doi or doi.lower().startswith('10.48550/'):
        # arXiv's DOI is registered with DataCite rather than Crossref.
        endpoint='https://api.datacite.org/dois/'+doi if doi else e['url']
    else: endpoint='https://api.crossref.org/works/'+doi
    record['endpoint']=endpoint
    try:
        r=requests.get(endpoint,timeout=35,headers={'User-Agent':'AcademicBibliographyVerification/1.0'})
        record['http_status']=r.status_code
        r.raise_for_status()
        if 'api.crossref' in endpoint:
            m=r.json()['message'];record['metadata']=m
            record['title']=m.get('title',[''])[0]
            record['status']='verified_registry'
        elif 'api.datacite' in endpoint:
            m=r.json()['data']['attributes'];record['metadata']=m
            record['title']=m.get('titles',[{}])[0].get('title','')
            record['status']='verified_registry'
        else:
            record['status']='verified_source_page'
        if 'title' in record:
            normalize=lambda t: re.sub('[^a-z0-9]','',html.unescape(t).lower())
            record['title_matches']=normalize(record['title'])==normalize(e['title'])
    except Exception as ex:
        record['status']='live_lookup_unavailable';record['error']=str(ex)
    return record

def main():
    entries={}
    source_bib=OUT/'references.bib' if (OUT/'references.bib').exists() else ROOT/'manuscript/references.bib'
    for p in [source_bib,ROOT/'literature/direct_neighbors_2026.bib']:
        entries.update({e['ID']:e for e in bibtexparser.loads(p.read_text(encoding='utf-8')).entries})
    chosen=[entries[k].copy() for k in KEYS]
    # Volume/issue publication year: the DOI contains the earlier online year.
    next(e for e in chosen if e['ID']=='ratnam2015ausgrid')['year']='2017'
    for e in chosen:e['title']='{'+e['title'].strip('{}')+'}'
    db=bibtexparser.bibdatabase.BibDatabase();db.entries=chosen
    OUT.mkdir(exist_ok=True);(OUT/'references.bib').write_text(bibtexparser.dumps(db),encoding='utf-8')
    with ThreadPoolExecutor(max_workers=5) as pool: records=list(pool.map(verify,chosen))
    (OUT/'source_data/reference_verification.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{'key':r['key'],'status':r['status'],'title_matches':r.get('title_matches')} for r in records],indent=2))

if __name__=='__main__': main()
