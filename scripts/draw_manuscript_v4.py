"""Five vector research figures, derived exclusively from v3 Source Data."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'manuscript_v4';S=O/'source_data';F=O/'figures';Q=O/'quality'
F.mkdir(exist_ok=True);Q.mkdir(exist_ok=True)
helper=ROOT/'scripts/figure_helpers'
sys.path.insert(0,str(helper) if helper.exists() else r'C:\Users\Lenovo\.codex\skills\nature-figure\scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'lines.linewidth':1.35,'savefig.facecolor':'white'})
W=183/25.4;BLUE='#176B8F';TEAL='#168C85';ORANGE='#C36A39';DARK='#23343F';GRAY='#737F87';LIGHT='#E9EEF1'
DSCOL={'eia930':BLUE,'ecl':ORANGE,'ausgrid':TEAL}
CM=LinearSegmentedColormap.from_list('effects',[BLUE,'#F7F8F8',ORANGE])
A=pd.read_csv(S/'all_method_metrics.csv');C=pd.read_csv(S/'paired_comparisons.csv');B=pd.read_csv(S/'paired_ablations.csv');SE=pd.read_csv(S/'seed_metrics.csv')
cells=[('eia930',24),('eia930',168),('ecl',24),('ecl',168),('ausgrid',24),('ausgrid',168)]
names={'eia930':'EIA-930','ecl':'ECL','ausgrid':'Ausgrid'}
clabels=[f'{names[d]} · {h} h' for d,h in cells]
sc=['normal','missingness','shock','controlled_drift','anomaly'];sl=['Original','Missing','Recent\nscale','Affine\noffset','Block\nanomaly']
def panel(ax,letter,title):
    ax.text(-.14,1.07,letter,transform=ax.transAxes,weight='bold',fontsize=11,color=DARK,va='bottom')
    ax.set_title(title,loc='left',pad=13,color=DARK)
def grid(ax,axis='x'):
    ax.grid(axis=axis,color=LIGHT,linewidth=.65);ax.set_axisbelow(True)
def save(fig,name):
    fig.canvas.draw()
    axes=[a for a in fig.axes if a.get_label()!='colorbar']
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list('abcdefghijklmnopqrstuvwxyz')[:len(axes)],json_out=Q/(name+'_alignment.json'),overlay_svg=Q/(name+'_alignment.svg'))
    for ext in ['pdf','svg','png','tiff']:
        opts={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
        fig.savefig(F/f'{name}.{ext}',dpi=600,**opts)
    plt.close(fig)
def point(ax,y,row,color=BLUE):
    v,lo,hi=100*np.array([row.relative_difference,row.relative_ci_low,row.relative_ci_high])
    ax.errorbar(v,y,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=color,ms=4.5,capsize=2,lw=1.2)

# 1. Architecture and chronological boundary: a schematic, not an empirical plot.
fig,ax=plt.subplots(figsize=(W,4.85));fig.subplots_adjust(left=.025,right=.985,bottom=.025,top=.97)
ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
def txt(x,y,t,**kw):
    kw.setdefault('color',DARK);ax.text(x,y,t,va='center',**kw)
def box(x,y,w,h,title,body='',fc='#F0F4F6',edge='#CCD7DE'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.5,rounding_size=1.4',fc=fc,ec=edge,lw=.8))
    txt(x+w/2,y+h-4,title,ha='center',fontsize=8,weight='bold')
    if body:txt(x+w/2,y+h/2-2.5,body,ha='center',fontsize=7,linespacing=1.5)
def arrow(x1,y1,x2,y2,color=GRAY):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=1,color=color))
txt(1,97,'a',weight='bold',fontsize=11);txt(5,97,'Forecast first; score when observations arrive',fontsize=10,weight='bold')
for x,w,label,fc in [(5,48,'336 h observed context','#E7F0F4'),(57,38,'24 h or 168 h forecast','#DFF0EC')]:
    ax.add_patch(FancyBboxPatch((x,84),w,7,boxstyle='round,pad=0,rounding_size=1',fc=fc,ec='none'));txt(x+w/2,87.5,label,ha='center',fontsize=8)
ax.plot([55,55],[81,93],color=DARK,lw=1,ls='--');txt(55,78,'Forecast origin',ha='center',fontsize=7)
txt(1,71,'b',weight='bold',fontsize=11);txt(5,71,'Separate the seasonal reference, learned correction and online state',fontsize=10,weight='bold')
box(2,45,19,17,'Hourly context','Load + calendar\nTraining-scale units')
box(27,45,21,17,'Context conditioning','25 h median filter\nFixed clipping')
box(55,46,20,17,'Frozen encoder','4 residual TCN levels\n128 features')
box(80,46,18,17,'Low-rank state','z → z + UVz\n2,048 parameters',fc='#DFF0EC',edge=TEAL)
arrow(21.5,54,26.4,54);arrow(48.6,54,54.3,54);arrow(75.5,54,79.4,54)
box(27,20,21,17,'Seasonal reference','Last day + 7-day mean')
box(55,20,43,17,'Additive forecast','Reference + 0.5 × (linear + neural correction)')
arrow(38,44.5,38,37.5);arrow(48.5,28.5,54.5,28.5);arrow(89,45.5,89,37.5);arrow(65,45.5,65,37.5)
txt(3,31,'Only U and V\nchange online',fontsize=8,color=TEAL,weight='bold',linespacing=1.7)
ax.plot([5,95],[15,15],color=LIGHT,lw=1)
txt(5,9,'Context drift > validation threshold',fontsize=7.5)
arrow(44,9,49,9)
txt(51,9,'One masked-reconstruction step · clipped and projected',fontsize=7.5,color=TEAL)
save(fig,'fig1_architecture')

# 2. Chronological paired effects, both accuracy metrics and every main cell.
fig,axs=plt.subplots(1,2,figsize=(W,3.65));fig.subplots_adjust(left=.17,right=.97,bottom=.18,top=.82,wspace=.46)
for ax,metric,letter in zip(axs,['mae','rmse'],'ab'):
    panel(ax,letter,metric.upper()+' against validation comparator')
    ax.axvline(0,color=GRAY,lw=.8);grid(ax)
    for i,(d,h) in enumerate(cells):
        q=C[(C.dataset==d)&(C.horizon==h)&(C.scenario=='normal')&(C.metric==metric)&(C.block_days==7)].iloc[0]
        point(ax,5-i,q,DSCOL[d])
    ax.set(yticks=range(6),yticklabels=clabels[::-1],ylim=(-.6,5.6),xlim=(-20,30),xticks=[-20,-10,0,10,20,30],xlabel='StableShift − comparator (%)')
axs[1].set_yticklabels([])
fig.text(.5,.035,'Points: five-seed pooled errors   |   Bars: paired 95% time-block bootstrap intervals',ha='center',fontsize=7,color=GRAY)
save(fig,'fig2_accuracy')

# 3. Relative response and absolute comparative performance under input stress.
D=pd.read_csv(S/'degradation_ratios.csv')
fig,axs=plt.subplots(2,2,figsize=(W,5.6));fig.subplots_adjust(left=.105,right=.97,bottom=.12,top=.88,wspace=.30,hspace=.79)
for ax,h,l in zip(axs[0],[24,168],'ab'):
    panel(ax,l,f'EIA-930 · {h} h stress response')
    for m,col,marker in [('TiDE',GRAY,'s'),('StableShift',BLUE,'o')]:
        q=D[(D.dataset=='eia930')&(D.horizon==h)&(D.method==m)].set_index('scenario').loc[sc]
        ax.plot(range(5),q.mae_ratio,marker=marker,ms=4,color=col,label=m)
    ax.axhline(1,color=LIGHT,lw=.8);ax.set(xticks=range(5),xticklabels=sl,ylabel='MAE / own original-input MAE',ylim=(.94,1.43));grid(ax,'y')
    if h==24:fig.legend(*ax.get_legend_handles_labels(),frameon=False,loc='upper right',bbox_to_anchor=(.97,.995),ncol=2)
for ax,h,l in zip(axs[1],[24,168],'cd'):
    panel(ax,l,f'{h} h comparative MAE change (%)')
    values=np.array([[100*C[(C.dataset==d)&(C.horizon==h)&(C.scenario==s)&(C.metric=='mae')&(C.block_days==7)].iloc[0].relative_difference for s in sc] for d in names])
    ax.imshow(values,aspect='auto',cmap=CM,norm=TwoSlopeNorm(vmin=-50,vcenter=0,vmax=50))
    for i in range(3):
        for j in range(5):ax.text(j,i,f'{values[i,j]:+.1f}',ha='center',va='center',fontsize=7,color='white' if abs(values[i,j])>32 else DARK)
    ax.set(xticks=range(5),xticklabels=sl,yticks=range(3),yticklabels=list(names.values()))
    ax.tick_params(length=0);ax.spines[['left','bottom']].set_visible(False)
fig.text(.5,.02,'Blue: lower error than comparator    ·    White: near zero    ·    Orange: higher error',ha='center',fontsize=7,color=GRAY)
save(fig,'fig3_input_response')

# 4. Matched component interventions. Small online effects keep their true scale.
fig,axs=plt.subplots(2,2,figsize=(W,5.9));fig.subplots_adjust(left=.17,right=.97,bottom=.10,top=.91,wspace=.43,hspace=.70)
for ax,v,l,title in [(axs[0,0],'anchor_only','a','Removing learned corrections'),(axs[0,1],'no_stability','b','Removing representation penalties')]:
    panel(ax,l,title);ax.axvline(0,color=GRAY,lw=.7)
    if v=='anchor_only':grid(ax)
    for i,(d,h) in enumerate(cells):
        q=B[(B.dataset==d)&(B.horizon==h)&(B.scenario=='normal')&(B.metric=='mae')&(B.variant==v)]
        if not q.empty:point(ax,5-i,q.iloc[0],DSCOL[d])
        else:ax.text(.98,(5-i+.6)/6.2,'n < 5',transform=ax.transAxes,ha='right',va='center',fontsize=6.5,color=GRAY)
    ax.set(yticks=range(6),yticklabels=clabels[::-1],ylim=(-.6,5.6),xlabel='MAE change from full model (%)')
axs[0,1].set_yticklabels([])
ax=axs[1,0];panel(ax,'c','Disabling online updates: MAE change (%)')
values=np.array([[100*B[(B.dataset==d)&(B.horizon==h)&(B.scenario==s)&(B.metric=='mae')&(B.variant=='no_update')].iloc[0].relative_difference for s in sc] for d,h in cells])
ax.imshow(values,aspect='auto',cmap=CM,norm=TwoSlopeNorm(vmin=-.06,vcenter=0,vmax=.06))
for i in range(6):
    for j in range(5):ax.text(j,i,f'{values[i,j]:+.3f}' if abs(values[i,j])>=.0005 else '0',ha='center',va='center',fontsize=6.3,color='white' if abs(values[i,j])>.037 else DARK)
ax.set(xticks=range(5),xticklabels=sl,yticks=range(6),yticklabels=clabels);ax.tick_params(length=0);ax.spines[['left','bottom']].set_visible(False)
ax=axs[1,1];panel(ax,'d','Updates in the chronological stream')
for i,(d,h) in enumerate(cells):
    q=A[(A.dataset==d)&(A.horizon==h)&(A.method=='StableShift')&(A.variant=='online')&(A.scenario=='normal')].iloc[0]
    y=5-i;frac=100*q.updates_mean/q.deployment_origins
    ax.barh(y,frac,color=DSCOL[d],height=.5)
    ax.text(frac+.7,y,f'{int(q.updates_mean)}/{q.deployment_origins}',va='center',fontsize=7,color=DARK)
ax.set(yticks=range(6),yticklabels=[],ylim=(-.6,5.6),xlim=(0,32),xlabel='Scheduled origins updated (%)')
save(fig,'fig4_components')

# 5. Held-out regions and hardware-specific deployment measurements.
U=pd.read_csv(S/'transfer_absolute_intervals.csv')
fig,axs=plt.subplots(2,2,figsize=(W,5.65));fig.subplots_adjust(left=.145,right=.97,bottom=.105,top=.91,wspace=.43,hspace=.70)
ml=['SeasonalNaive','WeeklyNaive','DLinear','PatchTST','StableShift'];short=['Daily','Weekly','DLinear','PatchTST','StableShift']
for ax,d,l in zip(axs[0],['eia930_ERCO','eia930_SWPP'],'ab'):
    panel(ax,l,d[7:]+' · held-out region, 168 h')
    for i,m in enumerate(ml):
        q=U[(U.dataset==d)&(U.method==m)].iloc[0];col=BLUE if m=='StableShift' else GRAY
        ax.errorbar(q['mean'],4-i,xerr=[[q['mean']-q.ci_low],[q.ci_high-q['mean']]],fmt='o',color=col,ms=4.5,lw=1.2,capsize=2)
    ax.set(yticks=range(5),yticklabels=short[::-1],ylim=(-.6,4.6),xlabel='MAE (MW)');grid(ax)
ax=axs[1,0];panel(ax,'c','EIA-930 · 24 h model size')
mm=['DLinear','PatchTST','TimesNet','iTransformer','TiDE','StableShift']
q=A[(A.dataset=='eia930')&(A.horizon==24)&(A.scenario=='normal')&A.method.isin(mm)&A.variant.isin(['frozen','online'])]
for r in q.itertuples():
    col=BLUE if r.method=='StableShift' else GRAY
    ax.scatter(r.parameters/1e6,r.mae,s=25,color=col)
    offset={'DLinear':(6,0),'PatchTST':(-37,-23),'iTransformer':(2,16),'TiDE':(18,0),'TimesNet':(-48,7),'StableShift':(5,-12)}[r.method]
    arrowprops={'arrowstyle':'-','color':col,'lw':.55} if r.method in ['PatchTST','TiDE','iTransformer'] else None
    ax.annotate(r.method,(r.parameters/1e6,r.mae),xytext=offset,textcoords='offset points',fontsize=7,color=col,arrowprops=arrowprops)
ax.set_xscale('log');ax.set(xlabel='Trainable parameters (millions)',ylabel='MAE (MW)',ylim=(1480,2020));ax.margins(x=.3);grid(ax,'y')
ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
ax=axs[1,1];panel(ax,'d','EIA-930 · 24 h forecast latency')
for i,m in enumerate(mm):
    rows=SE[(SE.dataset=='eia930')&(SE.horizon==24)&(SE.scenario=='normal')&(SE.method==m)&SE.variant.isin(['online','frozen'])]
    v=rows.latency_median_ms.median();col=BLUE if m=='StableShift' else GRAY
    ax.plot([rows.latency_median_ms.min(),rows.latency_median_ms.max()],[5-i,5-i],color=col,lw=1.1)
    ax.scatter(v,5-i,color=col,s=20)
ax.set(yticks=range(6),yticklabels=mm[::-1],ylim=(-.6,5.6),xlabel='Median per-origin latency (ms)',xscale='log');grid(ax)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
save(fig,'fig5_transfer_cost')

print('Exported five figures as PDF, SVG, PNG and LZW TIFF (600 dpi).')
