"""Publication figures from complete, observed strengthening results only.

Style inheritance: the author's original forest-plot palette and typography.
Data mapping and statistical families are rebuilt for the expanded study.
See FIGURE_CONTRACT.md for the scientific and export contract.
"""
from pathlib import Path
import argparse,json,sys,subprocess,os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,NullFormatter

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
A=HERE/'analysis'
F=Path(os.environ.get('QNC_FIGURE_DIR',str(HERE/'figures')))
Q=Path(os.environ.get('QNC_FIGURE_QA_DIR',str(HERE/'figure_quality')))
DISPLAY_METHOD=os.environ.get('QNC_DISPLAY_METHOD','QNC')
F.mkdir(exist_ok=True);Q.mkdir(exist_ok=True)
HELPER=HERE/'figure_helpers'
SKILL=HELPER
sys.path.insert(0,str(HELPER))
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':8,
 'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.65,'savefig.facecolor':'white',
 'legend.frameon':False})
BLUE='#176B8F';TEAL='#168C85';ORANGE='#CA7547';GRAY='#7B858B';DARK='#263C45';LIGHT='#E6ECEE'
DS={'eia930':'EIA','ecl':'ECL','ausgrid':'Ausgrid'}
COLORS={'eia930':BLUE,'ecl':TEAL,'ausgrid':ORANGE}
CELLS=[(ds,h,b) for ds in DS for h in [24,168] for b in ['DLinear','StableShift']]
NEWCELLS=[c for c in CELLS if c[0]!='eia930']
LABELS=lambda cells:[f'{DS[ds]} · {b} · {h} h' for ds,h,b in cells]
MANIFEST={}

def panel(ax,letter,title):
    ax.text(-.07,1.075,letter,transform=ax.transAxes,fontsize=11,weight='bold',color=DARK,va='bottom')
    ax.set_title(title,loc='left',pad=15,color=DARK)

def save(fig,axes,name,sources):
    if os.environ.get('QNC_BRITISH_SPELLING')=='1':
        from matplotlib.text import Text
        for artist in fig.findobj(match=Text):
            value=artist.get_text()
            artist.set_text(value.replace('Realized','Realised').replace('Gray:','Grey:').replace('Color:','Colour:').replace(' favor ',' favour '))
    fig.canvas.draw()
    axs=list(np.asarray(axes).ravel())
    require_matplotlib_panel_alignment(fig,axes=axs,panel_ids=list('abcdef')[:len(axs)],
        json_out=Q/f'{name}_alignment.json',overlay_svg=Q/f'{name}_alignment.svg',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(F/f'{name}.pdf')
    fig.savefig(F/f'{name}.svg')
    fig.savefig(F/f'{name}.png',dpi=600)
    fig.savefig(F/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    MANIFEST[name]={'sources':sources,'width_mm':183,'raster_dpi':600,
        'all_predeclared_settings_retained':True,'missing_or_synthetic_data':False}
    for script,args in [('audit_pdf_text.py',['--min-pt','5','--json']),
        ('audit_figure_collisions.py',['--json-out',str(Q/f'{name}_collision.json'),'--overlay-pdf',str(Q/f'{name}_collision.pdf'),'--json'])]:
        tool=(HELPER/script) if (HELPER/script).exists() else SKILL/script
        proc=subprocess.run([sys.executable,str(tool),str(F/f'{name}.pdf'),*args],capture_output=True,text=True)
        (Q/f'{name}_{script}.txt').write_text(proc.stdout+proc.stderr,encoding='utf-8')
        if proc.returncode:raise RuntimeError(f'{name} {script}: '+(proc.stdout+proc.stderr)[-5000:])

def forest(ax,frame,cells,show_labels=True,color=BLUE):
    n=len(cells)
    for i,(ds,h,b) in enumerate(cells):
        row=frame[(frame.dataset==ds)&(frame.horizon==h)&(frame.base==b)]
        assert len(row)==1,(ds,h,b,len(row))
        r=row.iloc[0];x,lo,hi=100*np.array([r.relative_difference,r.relative_ci_low,r.relative_ci_high])
        assert np.isfinite([x,lo,hi]).all() and lo<=hi
        y=n-1-i
        ax.plot([lo,hi],[y,y],color=color,lw=1.15,zorder=2)
        ax.plot([lo,lo],[y-.08,y+.08],color=color,lw=.75)
        ax.plot([hi,hi],[y-.08,y+.08],color=color,lw=.75)
        ax.plot(x,y,'o',ms=4,mec=color,mfc=color if r.holm_p<.05 else 'white',mew=1.05,zorder=3)
    ax.set_ylim(-.65,n-.35);ax.set_yticks(np.arange(n))
    ax.set_yticklabels(LABELS(cells)[::-1] if show_labels else ['']*n)
    ax.tick_params(axis='y',length=0,pad=5)
    if n==12:ax.axhspan(3.5,7.5,color='#F5F7F8',zorder=-3)
    ax.axvline(0,color=GRAY,lw=.75);ax.grid(axis='x',color=LIGHT,lw=.6);ax.set_axisbelow(True)
    low=min(-.25,100*frame.relative_ci_low.min());high=max(.25,100*frame.relative_ci_high.max())
    pad=(high-low)*.07;ax.set_xlim(low-pad,high+pad)
    ax.set_xlabel('MAE difference (%)')

def comparisons(phase,family,baseline):
    p=pd.read_csv(A/f'{phase}_paired_comparisons.csv')
    return p[(p.family==family)&(p.baseline==baseline)&(p.metric=='mae')&(p.block_days==7)]

def accuracy():
    fig,axs=plt.subplots(1,3,figsize=(7.2047244,4.1732283))
    fig.subplots_adjust(left=.205,right=.975,top=.86,bottom=.17,wspace=.38)
    for j,(baseline,title) in enumerate([('frozen','Frozen source'),('cosa_expanded','Expanded COSA'),('nlms_expanded','Expanded NLMS')]):
        panel(axs[j],'abc'[j],title);forest(axs[j],comparisons('test','primary',baseline),CELLS,j==0)
    fig.text(.205,.04,f'Negative values favor {DISPLAY_METHOD}. Filled points: Holm-adjusted p < 0.05.',fontsize=7,color=GRAY)
    save(fig,axs,'fig2_expanded_accuracy',['test_paired_comparisons.csv'])

def mechanism():
    fig,axs=plt.subplots(2,2,figsize=(7.2047244,5.7874016))
    fig.subplots_adjust(left=.205,right=.975,top=.91,bottom=.10,wspace=.72,hspace=.66)
    panel(axs[0,0],'a','Matched-step NLMS')
    forest(axs[0,0],comparisons('test','matched_mechanism','nlms_matched_new'),CELLS)
    panel(axs[0,1],'b','Matched historical clipping')
    forest(axs[0,1],comparisons('test','matched_mechanism','history_matched_new'),CELLS,False)
    panel(axs[1,0],'c','Effect on the current query')
    g=pd.read_csv(A/'test_update_geometry.csv')
    event_keys=['dataset','horizon','base','seed','update_number']
    reference=g[g.method=='query_expanded'][event_keys+['query_cap']].rename(columns={'query_cap':'reference_query_cap'})
    audit=[]
    for name,color,label in [('query_expanded',BLUE,'QNC'),('history_matched_new',TEAL,'Historical clipping'),('nlms_matched_new',ORANGE,'NLMS')]:
        rows=g[g.method==name].merge(reference,on=event_keys,validate='one_to_one')
        assert (rows.reference_query_cap>0).all()
        values=(rows.query_rms/rows.reference_query_cap).to_numpy();assert np.isfinite(values).all() and (values>=0).all()
        values=np.sort(values);axs[1,0].step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,lw=1.4,label=label,zorder=4 if name=='query_expanded' else 2)
        audit.append({'method':name,'updates':len(values),'max_query_ratio':float(values.max()),'p99_query_ratio':float(np.quantile(values,.99))})
    axs[1,0].axvline(1,color=GRAY,ls='--',lw=.8,zorder=1)
    # Show every displacement, including the unbounded comparator tail.
    limit=max(1.05,max(r['max_query_ratio'] for r in audit)*1.03)
    if limit>5:axs[1,0].set_xscale('symlog',linthresh=1)
    axs[1,0].set(xlim=(0,limit),ylim=(0,1.03),xlabel='Current RMS change / query budget',ylabel='Fraction of active updates')
    axs[1,0].xaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
    axs[1,0].xaxis.set_minor_formatter(NullFormatter())
    axs[1,0].legend(loc='lower right',fontsize=7)
    axs[1,0].grid(axis='y',color=LIGHT,lw=.6);axs[1,0].set_axisbelow(True)
    (Q/'geometry_display.json').write_text(json.dumps(audit,indent=2))
    panel(axs[1,1],'d','Unbatched calibration and scoring')
    runtime=pd.read_csv(A/'runtime_summary.csv').pivot(index=['dataset','horizon','base'],columns='method',values='ms_per_origin')
    repetition=pd.read_csv(A/'runtime_repetitions.csv')
    active=repetition.groupby(['dataset','horizon','base','method']).updates.max().unstack()
    use=(active.query_expanded>0)&(active.cosa_expanded>0)
    selected=runtime[use];assert len(selected)>0
    x=selected.cosa_expanded;y=selected.query_expanded
    for ds in DS:
        ix=selected.index.get_level_values('dataset')==ds
        axs[1,1].scatter(x[ix],y[ix],s=20,color=COLORS[ds],edgecolor='white',linewidth=.3,zorder=3,label=DS[ds])
    lower=min(x.min(),y.min())*.65;upper=max(x.max(),y.max())*1.4
    axs[1,1].plot([lower,upper],[lower,upper],color=GRAY,lw=.8,ls='--')
    axs[1,1].set(xscale='log',yscale='log',xlim=(lower,upper),ylim=(lower,upper),xlabel='Controlled COSA (ms/origin)',ylabel='QNC (ms/origin)')
    for axis in [axs[1,1].xaxis,axs[1,1].yaxis]:
        axis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'));axis.set_minor_formatter(NullFormatter())
    axs[1,1].legend(loc='upper left',fontsize=7,handlelength=.8)
    (Q/'runtime_display.json').write_text(json.dumps({'seed':11,'available_settings':len(runtime),'displayed_active_settings':len(selected),'median_cosa_to_qnc_ratio':float((x/y).median()),'excludes_common_source_inference':True},indent=2))
    save(fig,axs,'fig3_controls',['test_paired_comparisons.csv','test_update_geometry.csv','runtime_summary.csv','runtime_repetitions.csv'])

def budget():
    m=pd.read_csv(A/'test_aggregate_metrics.csv')
    g=pd.read_csv(A/'test_geometry_summary.csv')
    frozen=m[m.method=='frozen'].set_index(['dataset','horizon','base'])
    rows=[]
    for _,r in m[m.method.str.startswith('budget_')].iterrows():
        strategy,rho=r.method.removeprefix('budget_').split('_');rho=float(rho)
        geo=g[(g.dataset==r.dataset)&(g.horizon==r.horizon)&(g.base==r.base)&(g.method==r.method)]
        rows.append({'dataset':r.dataset,'horizon':r.horizon,'base':r.base,'strategy':strategy,'rho':rho,
          'relative_mae':100*(r.mae/frozen.loc[(r.dataset,r.horizon,r.base),'mae']-1),
          'active_updates':0 if len(geo)==0 else int(geo.iloc[0].updates),
          'query_rms_p99':np.nan if len(geo)==0 else float(geo.iloc[0].query_rms_p99),
          'clipped_fraction':np.nan if len(geo)==0 else float(geo.iloc[0].clipped_fraction)})
    data=pd.DataFrame(rows);assert len(data)==120
    data.to_csv(A/'budget_profiles.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(7.2047244,3.1102362))
    fig.subplots_adjust(left=.09,right=.97,top=.81,bottom=.24,wspace=.50)
    for j,(strategy,title) in enumerate([('fixed','Fixed primary step size'),('tuned','Validation-selected step size')]):
        panel(axs[j],'abc'[j],title);part=data[data.strategy==strategy]
        for _,r in part.groupby(['dataset','horizon','base']):axs[j].plot(r.rho,r.relative_mae,color='#C5CDD0',lw=.6,alpha=.7,zorder=1)
        for ds,color in COLORS.items():
            r=part[part.dataset==ds].groupby('rho').relative_mae.mean()
            axs[j].plot(r.index,r.values,'o-',ms=3,lw=1.5,color=color,label=DS[ds],zorder=3)
        axs[j].axhline(0,color=GRAY,lw=.7);axs[j].set_ylabel('MAE versus frozen (%)' if j==0 else '')
    panel(axs[2],'c','Realized current displacement')
    part=data[data.strategy=='fixed']
    for _,r in part.groupby(['dataset','horizon','base']):axs[2].plot(r.rho,r.query_rms_p99,color='#C5CDD0',lw=.6,alpha=.7)
    for ds,color in COLORS.items():
        r=part[part.dataset==ds].groupby('rho').query_rms_p99.mean()
        axs[2].plot(r.index,r.values,'o-',ms=3,lw=1.5,color=color)
    axs[2].set_ylabel('99th-percentile RMS change')
    for ax in axs:
        ax.set(xscale='log',xlabel='Query budget ρ',xticks=[.01,.025,.05,.1,.2])
        ax.set_xticklabels(['.01','.025','.05','.1','.2']);ax.grid(axis='y',color=LIGHT,lw=.6);ax.set_axisbelow(True)
    axs[0].legend(loc='upper left',bbox_to_anchor=(-.05,-.25),ncol=3,columnspacing=.8,handlelength=1,fontsize=7)
    fig.text(.45,.035,'Gray: all settings. Color: dataset means.',fontsize=7,color=GRAY)
    save(fig,axs,'fig4_budget',['test_aggregate_metrics.csv','test_geometry_summary.csv','budget_profiles.csv'])

def transfer():
    fig,axs=plt.subplots(1,3,figsize=(7.2047244,3.6614173))
    fig.subplots_adjust(left=.205,right=.975,top=.84,bottom=.19,wspace=.38)
    for j,(family,baseline,title) in enumerate([('primary','frozen','Frozen source'),('primary','nlms_expanded','Fixed NLMS'),('independently_tuned_history','history_clip','Fixed historical clipping')]):
        panel(axs[j],'abc'[j],title)
        forest(axs[j],comparisons('withheld',family,baseline),NEWCELLS,j==0)
    fig.text(.205,.045,f'Configurations fixed before new entities were scored. Negative values favor {DISPLAY_METHOD}.',fontsize=7,color=GRAY)
    save(fig,axs,'fig5_entity_transfer',['withheld_paired_comparisons.csv'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figure',choices=['all','accuracy','mechanism','budget','transfer'],default='all');args=parser.parse_args()
    choices={'accuracy':accuracy,'mechanism':mechanism,'budget':budget,'transfer':transfer}
    for name,method in choices.items():
        if args.figure in ['all',name]:method()
    (Q/'figure_manifest.json').write_text(json.dumps(MANIFEST,indent=2))
    proc=subprocess.run([sys.executable,str(SKILL/'validate_figure.py'),str(Path(__file__)),'--backend','python','--json'],capture_output=True,text=True)
    (Q/'source_validation.json').write_text(proc.stdout,encoding='utf-8')
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    print('SELECTED FIGURES EXPORTED AND AUDITED',args.figure,flush=True)
