from pathlib import Path
import subprocess,sys

HERE=Path(__file__).resolve().parent
runner=HERE/'run_strengthening.py'
def stage(name):
    while True:
        code=subprocess.call([sys.executable,str(runner),'--stage',name],cwd=HERE.parent)
        if code==75:continue
        if code:raise SystemExit(code)
        break
for name in ['select','lock','test','independent']:stage(name)
if not (HERE/'withheld_data_manifest.json').exists():
    subprocess.run([sys.executable,str(HERE/'prepare_withheld_entities.py')],cwd=HERE.parent,check=True)
stage('withheld')
print('ALL PREDECLARED EXPERIMENTS COMPLETE',flush=True)
