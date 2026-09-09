from pathlib import Path
import subprocess,os,json,re,shutil
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'manuscript_v4'
(O/'quality').mkdir(parents=True,exist_ok=True)
TEX=Path(r'F:\CCFA\texlive_link\bin\windows')
env=os.environ.copy()
if TEX.exists(): env['PATH']=str(TEX)+os.pathsep+env['PATH']
pdflatex=shutil.which('pdflatex',path=env['PATH']);bibtex=shutil.which('bibtex',path=env['PATH'])
if not pdflatex or not bibtex: raise SystemExit('Install TeX Live or MiKTeX and add pdflatex/bibtex to PATH.')
for name,bib in [('main',True),('supplementary',False),('cover_letter',False)]:
    if not (O/(name+'.tex')).exists():continue
    commands=[[pdflatex,'-interaction=nonstopmode','-halt-on-error',name+'.tex']]
    if bib:commands.append([bibtex,name])
    commands.extend([[pdflatex,'-interaction=nonstopmode','-halt-on-error',name+'.tex']]*2)
    for i,cmd in enumerate(commands):
        r=subprocess.run(cmd,cwd=O,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (O/'quality'/f'{name}_compile_{i}.txt').write_bytes(r.stdout)
        if r.returncode:
            print(r.stdout.decode('utf-8',errors='replace')[-5000:]);raise SystemExit(r.returncode)
    log=(O/(name+'.log')).read_text(encoding='utf-8',errors='replace')
    print(name,':',re.findall(r'Output written on[^\n]+',log))
    print('Overfull boxes:',len(re.findall('Overfull',log)))
print('Compiled main and supplement.')
