"""Final document, bibliography and figure checks; no experimental reruns."""
from pathlib import Path
import argparse,json,re,subprocess,sys
import fitz
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'manuscript_v4';Q=O/'quality'
Q.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--main-pdf',type=Path,default=O/'main.pdf',help='Main PDF to inspect when an open viewer requires an alternate export name.')
args=parser.parse_args()
checks={};text=(O/'main.tex').read_text(encoding='utf-8')
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',text,re.S).group(1)
checks['abstract_words']=len(abstract.split())
checks['references']=len(re.findall(r'\\bibitem', (O/'main.bbl').read_text(encoding='utf-8')))
cited=set(k for group in re.findall(r'\\cite\{([^}]+)\}',text) for k in group.split(','))
listed=set(re.findall(r'@\w+\{([^,]+),',(O/'references.bib').read_text(encoding='utf-8')))
checks['uncited_entries']=sorted(listed-cited);checks['missing_references']=sorted(cited-listed)
checks['highlight_lengths']=[len(x) for x in (O/'highlights.txt').read_text(encoding='utf-8').splitlines() if x.strip()]
checks['documents']={}
for name in ['main','supplementary','cover_letter']:
    pdf_path=args.main_pdf if name=='main' else O/(name+'.pdf')
    if not pdf_path.exists():continue
    log_path=O/(name+'.log')
    d=fitz.open(pdf_path);log=log_path.read_text(encoding='utf-8',errors='replace') if log_path.exists() else ''
    bad=[]
    for n,p in enumerate(d):
        for b in p.get_text('dict')['blocks']:
            for line in b.get('lines',[]):
                for span in line['spans']:
                    box=fitz.Rect(span['bbox'])
                    if box.x0 < -1 or box.x1 > p.rect.width+1 or box.y0 < -1 or box.y1 > p.rect.height+1:bad.append({'page':n+1,'text':span['text']})
    checks['documents'][name]={'pages':len(d),'overfull_boxes':len(re.findall('Overfull',log)) if log_path.exists() else None,
       'undefined_references':bool(re.search(r'(undefined references|Citation .* undefined|Reference .* undefined)',log)) if log_path.exists() else None,
       'text_outside_page':bad}
    for n in range(len(d)):d[n].get_pixmap(matrix=fitz.Matrix(1.35,1.35)).save(Q/f'{name}_page_{n+1:02}.png')
skill=Path(r'C:\Users\Lenovo\.codex\skills\nature-figure\scripts')
checks['figures']={}
for p in sorted((O/'figures').glob('*.pdf')):
    one={}
    for script,args,key in [('audit_pdf_text.py',['--min-pt','5','--json'],'text'),('audit_figure_collisions.py',['--json'],'collisions')]:
        if not (skill/script).exists():
            one[key]={'status':'not_run','reason':'Optional Nature-figure audit tool is not installed.'}
            continue
        r=subprocess.run([sys.executable,str(skill/script),str(p),*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try: one[key]=json.loads(r.stdout.decode('utf-8'))
        except Exception:one[key]={'exit_code':r.returncode,'output':r.stdout.decode('utf-8',errors='replace')[-1500:]}
        one[key]['process_exit_code']=r.returncode
    checks['figures'][p.stem]=one
(Q/'final_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in checks.items() if k!='figures'},indent=2))
print('Figure outputs:',{k:{kk:vv.get('process_exit_code') for kk,vv in v.items()} for k,v in checks['figures'].items()})
