"""Vector workflow schematic: no empirical observations are generated here."""
from pathlib import Path
import sys,os,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('QNC_SUBMISSION_DIR', str(ROOT / 'submission_eswa_nature_polished_20261010')))
(OUT / 'quality').mkdir(parents=True, exist_ok=True)
FIGURES = OUT / 'source' / 'figures'
sys.path.insert(0, str(ROOT / 'scripts' / 'figure_helpers'))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({'font.family':'Arial', 'font.size':8,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'savefig.facecolor':'white'})
BLUE, TEAL, ORANGE, DARK, GRAY = '#176B8F', '#168C85', '#CA7547', '#263C45', '#7B858B'

def box(ax, x, y, w, h, text, color=BLUE, fill='#EDF4F5', size=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.018',
                             fc=fill,ec=color,lw=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=color,fontsize=size,linespacing=1.4)

def arrow(ax, start, end, color=GRAY, width=1.1):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,color=color,lw=width))

fig, axes = plt.subplots(1,3,figsize=(7.2047,2.65))
fig.subplots_adjust(left=.035,right=.975,bottom=.12,top=.83,wspace=.29)
for ax in axes:
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.axis('off')
for ax, label, title in zip(axes, 'abc',
        ['A fixed forecasting service', 'Learn from delayed outcomes', 'Constrain the current revision']):
    ax.text(-.04,1.08,label,transform=ax.transAxes,fontsize=11,weight='bold',color=DARK)
    ax.text(.03,1.08,title,transform=ax.transAxes,fontsize=8.2,color=DARK,va='baseline')

ax=axes[0]
box(ax,.05,.73,.90,.20,'Observed load + calendar\nPast 336 h',GRAY,'#F5F7F8')
arrow(ax,(.50,.71),(.50,.55))
box(ax,.12,.35,.76,.20,'Frozen source model',BLUE)
arrow(ax,(.50,.33),(.50,.25),BLUE)
box(ax,.05,.04,.90,.19,'Source forecast\nNext 24 or 168 h',BLUE)
ax.text(.5,-.105,'Daily forecast issuance',ha='center',fontsize=7.2,color=DARK)

ax=axes[1]
box(ax,.02,.72,.43,.22,'Earlier forecast\nObserved load',TEAL,'#EDF5F2',7.5)
box(ax,.55,.72,.43,.22,'Current forecast\nObserved history',BLUE,size=7.5)
arrow(ax,(.235,.70),(.34,.46),TEAL)
arrow(ax,(.765,.70),(.66,.46),BLUE)
ax.text(.02,.56,'Residual',fontsize=7.2,color=TEAL)
ax.text(.73,.56,'Query',fontsize=7.2,color=BLUE)
box(ax,.09,.23,.82,.23,'Normalised proposal\n+ query-dependent step',BLUE,size=7.7)
arrow(ax,(.50,.21),(.50,.075),BLUE)
ax.text(.50,.025,'Update the output operator',ha='center',fontsize=7.7,color=DARK)
ax.text(.50,-.105,'Use each matured window once',ha='center',fontsize=7.2,color=DARK)

ax=axes[2]
centre=np.array([.34,.39]);end=np.array([.88,.83])
edge=centre+.285*(end-centre)/np.linalg.norm(end-centre)
ax.add_patch(Circle(centre,.285,fc='#EFF5F6',ec=BLUE,lw=1.1))
arrow(ax,centre,end,ORANGE,1.5)
arrow(ax,centre,edge,BLUE,2.2)
ax.plot(*centre,'o',color=DARK,ms=4)
ax.text(.31,.015,'Before update',ha='center',fontsize=7.5,color=DARK)
ax.text(.82,.91,'Proposal',ha='center',fontsize=7.5,color=ORANGE)
ax.text(.27,.80,'Allowed\nRMS change',ha='center',fontsize=7.5,color=BLUE,linespacing=1.3)
ax.text(.82,.40,'QORC\nrevision',ha='center',fontsize=7.5,color=BLUE,linespacing=1.3)
ax.text(.50,-.105,'Issue the corrected forecast once',ha='center',fontsize=7.2,color=DARK)

fig.canvas.draw()
require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list('abc'),
    json_out=OUT/'quality'/'fig1_method_alignment.json',
    overlay_svg=OUT/'quality'/'fig1_method_alignment.svg', tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
fig.savefig(FIGURES / 'fig1_method.pdf')
fig.savefig(FIGURES / 'fig1_method.svg')
fig.savefig(FIGURES / 'fig1_method.png', dpi=600)
fig.savefig(FIGURES / 'fig1_method.tiff', dpi=600, pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
print('Exported application workflow: PDF, SVG, PNG and TIFF.')

for script, args in [('audit_pdf_text.py', ['--min-pt','5','--json']), ('audit_figure_collisions.py', ['--json-out', str(OUT/'quality/fig1_method_collision.json'), '--overlay-pdf', str(OUT/'quality/fig1_method_collision.pdf'), '--json'])]:
    result = subprocess.run([sys.executable, str(ROOT/'research_strengthening_20261009/figure_helpers'/script), str(FIGURES/'fig1_method.pdf'), *args], capture_output=True, text=True)
    (OUT/'quality'/('fig1_method_'+script+'.txt')).write_text(result.stdout, encoding='utf-8')
    if result.returncode: raise RuntimeError(result.stdout+result.stderr)
