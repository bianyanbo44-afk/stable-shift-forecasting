"""Package the paper, code, and completed forecast evidence without changing runs."""
from pathlib import Path
import argparse,json,zipfile,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'manuscript_v4';OUT=ROOT/'submission_v4';OUT.mkdir(exist_ok=True)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--main-pdf',type=Path,default=O/'main.pdf',help='Use an alternate local export as manuscript_v4/main.pdf in the archive.')
parser.add_argument('--skip-evidence',action='store_true',help='Keep the existing completed-evidence archive and its manifest record.')
args=parser.parse_args()
if not args.main_pdf.is_file():raise SystemExit('Main PDF does not exist: '+str(args.main_pdf))
M=json.loads((O/'source_data/analysis_manifest.json').read_text(encoding='utf-8'))
def archive(name,paths,overrides=None):
    paths=sorted(set(p.resolve() for p in paths if p.is_file()))
    dest=OUT/name
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write((overrides or {}).get(p,p),p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(dest) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('CRC mismatch: '+bad)
    rec={'file':name,'files':len(paths),'bytes':dest.stat().st_size,'sha256':hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()}
    print(json.dumps(rec),flush=True);return rec
paper=[]
for pattern in ['*.tex','*.pdf','*.bib','*.bbl','*.txt','*.md','tables/*.tex','figures/*.pdf','figures/*.svg','figures/*.png','figures/*.tiff','source_data/*']:
    paper.extend(O.glob(pattern))
paper=[p for p in paper if p.name not in {'EDITORIAL_PLAN.md','main_final.pdf'} and (p.resolve()!=args.main_pdf.resolve() or p.name=='main.pdf')]
paper.append(O/'quality/final_checks.json')
records=[archive('Neurocomputing_manuscript_v4.zip',paper,{(O/'main.pdf').resolve():args.main_pdf.resolve()})]
code=list((ROOT/'src/stable_shift_forecasting').glob('*.py'))
script_names=['run_benchmark_v3.py','run_entity_transfer_v3.py','aggregate_benchmark_v3.py','analyze_manuscript_v4.py','build_submission_inventory_v4.py',
 'build_manuscript_assets_v4.py','draw_manuscript_v4.py','compile_manuscript_v4.py','review_manuscript_v4.py','prepare_references_v4.py',
 'download_data.py','download_recent_load.py','prepare_data.py','prepare_archival_v3.py','package_manuscript_v4.py']
code.extend(ROOT/'scripts'/s for s in script_names)
code.extend((ROOT/'scripts/figure_helpers').glob('*.py'))
code.extend(ROOT/'configs'/s for s in ['benchmark_v3.json','benchmark_v3_archival.json','entity_transfer_v3.json'])
code.extend(ROOT/s for s in ['pyproject.toml','requirements-lock.txt','requirements-cuda-lock.txt','requirements-upstream.txt','LICENSE','README.md','CITATION.cff','THIRD_PARTY_NOTICES.md','docs/REPRODUCIBILITY.md'])
code.extend(ROOT/'literature'/s for s in ['citation_verification_2026-09-08.json','crossref_2026_verified.json','direct_neighbors_2026.bib','neurocomputing_guide_2026-09-09.md'])
code.extend(ROOT/'data/prepared_v3'/s for s in ['ecl.json','ausgrid.json','pjm.json'])
code.extend(ROOT/'results'/s for s in ['data_manifest.json','eia930_2026_data_manifest.json'])
code.append(O/'README.md')
v=ROOT/'vendor_cache/Time-Series-Library'
for dirname in ['models','layers','utils']: code.extend((v/dirname).glob('*.py'))
code.extend([v/'LICENSE',v/'README.md'])
# Record the exact post-processing environment separately from experiment locks.
r=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True)
(OUT/'analysis_environment_lock.txt').write_bytes(r.stdout)
code.append(OUT/'analysis_environment_lock.txt')
records.append(archive('Neurocomputing_reproducibility_code_v4.zip',code))
evidence=[]
for record in M['sources']:
    evidence.extend([ROOT/record['path'],ROOT/record['prediction_path']])
roots=[ROOT/'results/benchmark_v3_formal',*[ROOT/'cloud_returned_20260909/results'/p for p in ['cloud_ecl_v3','cloud_ecl_h168_v3','cloud_ausgrid_v3','cloud_pjm_v3','benchmark_v3_transfer']]]
for r in roots:
    evidence.extend(r.rglob('*.json'))
    evidence.extend(r.rglob('training.csv'))
    evidence.extend((r/'source_snapshot').rglob('*.py'))
if args.skip_evidence:
    previous=json.loads((OUT/'package_manifest.json').read_text(encoding='utf-8'))
    saved=next(r for r in previous if r['file']=='Neurocomputing_completed_forecast_evidence_v4.zip')
    with (OUT/saved['file']).open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=saved['sha256']:raise RuntimeError('Existing evidence archive does not match its manifest.')
    records.append(saved)
else:
    records.append(archive('Neurocomputing_completed_forecast_evidence_v4.zip',evidence))
(OUT/'package_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
if (O/'quality/SELF_REVIEW.md').exists():
    (OUT/'SELF_REVIEW_INTERNAL.md').write_text((O/'quality/SELF_REVIEW.md').read_text(encoding='utf-8'),encoding='utf-8')
(OUT/'START_HERE.txt').write_text('Read manuscript_v4/main.pdf and manuscript_v4/supplementary.pdf.\nThe manuscript ZIP includes editable sources, figures, submission materials and Source Data.\nExtract all three archives into the SAME directory to preserve recorded project-relative paths.\nThe evidence ZIP preserves 2,120 completed scenario/variant forecast streams, predictions and protocol records.\nTraining weight checkpoints remain preserved in the original experiment folders and returned archive.\nNo training or external submission was started during manuscript production.\nSee manuscript_v4/README.md and quality/SELF_REVIEW.md before journal upload.\n',encoding='utf-8')
