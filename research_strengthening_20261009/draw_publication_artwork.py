"""Refine manuscript artwork from the complete, previously reported evidence.

Only visual encodings and layout change. No statistic is selected again.
PDF/SVG retain editable text; all observed settings and update events remain.
"""
from pathlib import Path
import json
import os
import sys
import subprocess
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.transforms import ScaledTranslation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = Path(os.environ.get('QNC_SUBMISSION_DIR', str(ROOT / 'submission_eswa_artwork_20261010')))
F = OUT / 'source/figures'
Q = OUT / 'quality/artwork'
Q.mkdir(parents=True, exist_ok=True)
F.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE / 'figure_helpers'))
from audit_panel_alignment import require_matplotlib_panel_alignment
import draw_strengthening as d

BLUE, ORANGE, PURPLE = '#176B8F', '#C46D3F', '#8064A2'
GRAY, DARK, LIGHT = '#68727B', '#20252B', '#B7BEC4'
DATASETS = {'eia930': (BLUE, 'o', '-'), 'ecl': (ORANGE, 's', '--'), 'ausgrid': (PURPLE, '^', ':')}
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 7, 'axes.titlesize': 7,
    'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5, 'text.color': DARK, 'axes.labelcolor': DARK,
    'axes.edgecolor': DARK, 'xtick.color': DARK, 'ytick.color': DARK,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
    'legend.frameon': False, 'savefig.facecolor': 'white', 'axes.grid': False,
    'xtick.major.size': 3, 'ytick.major.size': 3,
})
MANIFEST = json.loads((Q / 'figure_manifest.json').read_text(encoding='utf-8')) if (Q / 'figure_manifest.json').is_file() else {}

def panel(ax, letter, title):
    # Fixed point offsets keep panel anchors aligned at any axes height.
    transform = ax.transAxes + ScaledTranslation(-8/72, 9/72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, letter, transform=transform, fontsize=8, weight='bold', va='bottom')
    ax.set_title(title, loc='left', pad=11, color=DARK, fontsize=7)

def save(fig, axes, name, sources, details):
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(
        fig, axes=list(np.asarray(axes).ravel()), panel_ids=list('abcd')[:len(np.asarray(axes).ravel())],
        json_out=Q / f'{name}_alignment.json', overlay_svg=Q / f'{name}_alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    fig.savefig(F / f'{name}.pdf')
    fig.savefig(F / f'{name}.svg')
    fig.savefig(F / f'{name}.png', dpi=600)
    fig.savefig(F / f'{name}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    for script, args in [
        ('audit_pdf_text.py', ['--min-pt', '5', '--json']),
        ('audit_figure_collisions.py', ['--json-out', str(Q / f'{name}_collision.json'),
                                      '--overlay-pdf', str(Q / f'{name}_collision.pdf'), '--json'])]:
        result = subprocess.run([sys.executable, str(HERE / 'figure_helpers' / script),
                                 str(F / f'{name}.pdf'), *args], capture_output=True, text=True)
        (Q / f'{name}_{script}.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    MANIFEST[name] = {
        'width_mm': 183, 'height_mm': fig.get_size_inches()[1] * 25.4,
        'sources': sources, 'source_sha256': {p: hashlib.sha256((d.A / p).read_bytes()).hexdigest() for p in sources},
        'raster_dpi': 600, 'all_predeclared_settings_retained': True,
        'new_statistical_selection': False, 'alignment': alignment['verdict'], **details}
    (Q / 'figure_manifest.json').write_text(json.dumps(MANIFEST, indent=2) + '\n', encoding='utf-8')
    print('PASS:', name, flush=True)

def forest(ax, frame, cells, show_labels=True):
    d.forest(ax, frame, cells, show_labels, color=BLUE)
    ax.grid(False)
    ax.set_yticklabels(d.LABELS(cells)[::-1] if show_labels else [''] * len(cells), fontsize=6.5)
    ax.tick_params(axis='y', pad=4)
    ax.set_xlabel('QORC MAE difference (%)', fontsize=7)

def workflow():
    fig, axes = plt.subplots(1, 3, figsize=(183/25.4, 62/25.4))
    fig.subplots_adjust(left=.035, right=.975, bottom=.12, top=.80, wspace=.29)
    for ax, letter, title in zip(axes, 'abc', ['Frozen forecast', 'Delayed feedback', 'Query constraint']):
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.set_aspect('equal', adjustable='box')
        ax.axis('off'); panel(ax, letter, title)
    def box(ax, x, y, w, h, text, accent=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.012,rounding_size=.015',
                                   facecolor='#F5F7F8', edgecolor=accent, linewidth=.8))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=7, color=DARK, linespacing=1.35)
    def arrow(ax, start, end, color=GRAY, width=.8):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=7, color=color, linewidth=width))
    ax = axes[0]
    box(ax, .05, .73, .90, .20, 'Observed load + calendar\nPast 336 h', GRAY)
    arrow(ax, (.50, .71), (.50, .55))
    box(ax, .12, .35, .76, .20, 'Frozen source model')
    arrow(ax, (.50, .33), (.50, .25), BLUE)
    box(ax, .05, .04, .90, .19, 'Source forecast\nNext 24 or 168 h')
    ax.text(.5, -.11, 'Daily issuance', ha='center', fontsize=6.5)
    ax = axes[1]
    box(ax, .02, .72, .43, .22, 'Earlier forecast\nObserved load', GRAY)
    box(ax, .55, .72, .43, .22, 'Current forecast\nObserved history')
    arrow(ax, (.235, .70), (.34, .46), GRAY)
    arrow(ax, (.765, .70), (.66, .46), BLUE)
    ax.text(.02, .56, 'Residual', fontsize=6.5)
    ax.text(.73, .56, 'Query', fontsize=6.5)
    box(ax, .09, .23, .82, .23, 'Normalised proposal\n+ query-dependent step')
    arrow(ax, (.50, .21), (.50, .075), BLUE)
    ax.text(.50, .025, 'Update the output operator', ha='center', fontsize=7)
    ax.text(.50, -.11, 'Each matured window used once', ha='center', fontsize=6.5)
    ax = axes[2]
    centre, end = np.array([.34, .39]), np.array([.88, .83])
    edge = centre + .285 * (end-centre) / np.linalg.norm(end-centre)
    ax.add_patch(Circle(centre, .285, facecolor='#F2F6F8', edgecolor=BLUE, linewidth=.85))
    arrow(ax, centre, end, ORANGE, 1.2); arrow(ax, centre, edge, BLUE, 1.6)
    ax.plot(*centre, 'o', color=DARK, markersize=3)
    ax.text(.31, .015, 'Before update', ha='center', fontsize=6.5)
    ax.text(.82, .91, 'Proposal', ha='center', fontsize=6.5)
    ax.text(.27, .80, 'Allowed\nRMS change', ha='center', fontsize=6.5, linespacing=1.25)
    ax.text(.82, .40, 'QORC\nrevision', ha='center', fontsize=6.5, linespacing=1.25)
    ax.text(.50, -.11, 'Issue the corrected forecast', ha='center', fontsize=6.5)
    save(fig, axes, 'fig1_method', [], {'archetype': 'schematic-led composite', 'empirical_data': False})

def controls():
    fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 132/25.4))
    fig.subplots_adjust(left=.19, right=.975, top=.90, bottom=.10, wspace=.52, hspace=.55)
    control, timing, nlms, history = axes.ravel()
    panel(control, 'a', 'Current-query displacement')
    geometry = pd.read_csv(d.A / 'test_update_geometry.csv')
    keys = ['dataset', 'horizon', 'base', 'seed', 'update_number']
    reference = geometry[geometry.method == 'query_expanded'][keys + ['query_cap']].rename(columns={'query_cap': 'reference_cap'})
    audit = []
    for method, color, label, style in [
        ('query_expanded', BLUE, 'QORC', '-'),
        ('history_matched_new', GRAY, 'Historical clipping', '--'),
        ('nlms_matched_new', ORANGE, 'NLMS', ':')]:
        part = geometry[geometry.method == method].merge(reference, on=keys, validate='one_to_one')
        assert len(part) == len(reference) and (part.reference_cap > 0).all()
        values = np.sort((part.query_rms / part.reference_cap).to_numpy())
        assert np.isfinite(values).all() and (values >= 0).all()
        control.step(values, np.arange(1, len(values)+1)/len(values), where='post',
                     color=color, linestyle=style, linewidth=1.25,
                     label=label, zorder=4 if method == 'query_expanded' else 2)
        audit.append({'method': method, 'updates': len(values), 'ratio_max': float(values.max()),
                      'ratio_p99': float(np.quantile(values, .99))})
    control.axvline(1, color=GRAY, linestyle='--', linewidth=.6, zorder=1)
    control.set_xscale('symlog', linthresh=1)
    control.set(xlim=(0, max(r['ratio_max'] for r in audit)*1.03), ylim=(0, 1.03),
                xlabel='RMS change / query budget', ylabel='Fraction of active updates')
    control.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    control.xaxis.set_minor_formatter(NullFormatter())
    control.legend(loc='lower right', handlelength=2.2, labelspacing=.35)
    panel(timing, 'b', 'Calibration and scoring cost')
    summary = pd.read_csv(d.A / 'runtime_summary.csv').pivot(index=['dataset', 'horizon', 'base'], columns='method', values='ms_per_origin')
    reps = pd.read_csv(d.A / 'runtime_repetitions.csv')
    active = reps.groupby(['dataset', 'horizon', 'base', 'method']).updates.max().unstack()
    selected = summary[(active.query_expanded > 0) & (active.cosa_expanded > 0)]
    assert len(selected) == 11
    x, y = selected.cosa_expanded, selected.query_expanded
    for ds, (color, marker, style) in DATASETS.items():
        ix = selected.index.get_level_values('dataset') == ds
        timing.scatter(x[ix], y[ix], s=17, marker=marker, color=color, edgecolor='white',
                       linewidth=.35, label=d.DS[ds], zorder=3)
    lower, upper = min(x.min(), y.min())*.65, max(x.max(), y.max())*1.4
    timing.plot([lower, upper], [lower, upper], color=GRAY, linewidth=.65, linestyle='--')
    timing.set(xscale='log', yscale='log', xlim=(lower, upper), ylim=(lower, upper),
               xlabel='Controlled COSA (ms/origin)', ylabel='QORC (ms/origin)')
    for axis in (timing.xaxis, timing.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}')); axis.set_minor_formatter(NullFormatter())
    timing.legend(loc='upper left', handlelength=1, labelspacing=.4)
    timing.text(.65, .33, f'{(x/y).median():.1f}× median\nCOSA / QORC', transform=timing.transAxes,
                fontsize=6.5, ha='center', va='center', color=DARK)
    panel(nlms, 'c', 'Matched-step NLMS')
    forest(nlms, d.comparisons('test', 'matched_mechanism', 'nlms_matched_new'), d.CELLS)
    panel(history, 'd', 'Matched historical clipping')
    forest(history, d.comparisons('test', 'matched_mechanism', 'history_matched_new'), d.CELLS, False)
    save(fig, axes, 'fig2_control_first', ['test_update_geometry.csv', 'runtime_summary.csv',
         'runtime_repetitions.csv', 'test_paired_comparisons.csv'],
         {'geometry_events': audit, 'runtime_settings': 11, 'runtime_statistic': 'median of 3 technical repetitions',
          'runtime_uncertainty_exemption': 'Each point is the descriptive timing summary for a fixed seed11 setting; no inferential timing comparison is claimed.',
          'forest_uncertainty': 'pointwise paired 95% seed/calendar-block intervals; 7-day blocks',
          'forest_replicates': '5 seeds per each of 12 fixed settings'})

def accuracy():
    fig, axes = plt.subplots(1, 3, figsize=(183/25.4, 100/25.4))
    fig.subplots_adjust(left=.20, right=.975, top=.86, bottom=.17, wspace=.30)
    for i, (baseline, title) in enumerate([('frozen', 'Frozen source'), ('cosa_expanded', 'Controlled COSA'), ('nlms_expanded', 'Tuned NLMS')]):
        panel(axes[i], 'abc'[i], title)
        forest(axes[i], d.comparisons('test', 'primary', baseline), d.CELLS, i == 0)
    fig.text(.20, .04, 'Negative: lower MAE. Filled: Holm-adjusted p < 0.05.', fontsize=6.5)
    save(fig, axes, 'fig2_expanded_accuracy', ['test_paired_comparisons.csv'],
         {'settings_per_panel': 12, 'seeds_per_setting': 5, 'uncertainty': 'pointwise paired 95% seed/calendar-block intervals; 7-day blocks'})

def budget():
    data = pd.read_csv(d.A / 'budget_profiles.csv')
    assert len(data) == 120 and set(data.rho) == {.01, .025, .05, .1, .2}
    assert data.relative_mae.between(-12, 2).all(), 'Shared display limits must contain every score.'
    fig, axes = plt.subplots(1, 3, figsize=(183/25.4, 74/25.4))
    fig.subplots_adjust(left=.08, right=.98, top=.81, bottom=.23, wspace=.39)
    for i, (strategy, title) in enumerate([('fixed', 'Fixed primary step'), ('tuned', 'Validation-selected step')]):
        panel(axes[i], 'abc'[i], title)
        part = data[data.strategy == strategy]
        for _, rows in part.groupby(['dataset', 'horizon', 'base']):
            rows = rows.sort_values('rho'); axes[i].plot(rows.rho, rows.relative_mae, color=LIGHT, linewidth=.55, zorder=1)
        for ds, (color, marker, style) in DATASETS.items():
            rows = part[part.dataset == ds].groupby('rho').relative_mae.mean().sort_index()
            axes[i].plot(rows.index, rows.values, marker=marker, linestyle=style, markersize=3,
                         linewidth=1.2, color=color, label=d.DS[ds], zorder=3)
        axes[i].axhline(0, color=GRAY, linewidth=.65)
        axes[i].set(ylim=(-12, 2), yticks=[-10, -5, 0])
        axes[i].set_ylabel('MAE versus frozen (%)' if i == 0 else '')
    panel(axes[2], 'c', '99th-percentile displacement')
    part = data[data.strategy == 'fixed']
    for _, rows in part.groupby(['dataset', 'horizon', 'base']):
        rows = rows.sort_values('rho'); axes[2].plot(rows.rho, rows.query_rms_p99, color=LIGHT, linewidth=.55, zorder=1)
    for ds, (color, marker, style) in DATASETS.items():
        rows = part[part.dataset == ds].groupby('rho').query_rms_p99.mean().sort_index()
        axes[2].plot(rows.index, rows.values, marker=marker, linestyle=style, markersize=3, linewidth=1.2, color=color, zorder=3)
    axes[2].set_ylabel('RMS change (standardised)')
    for ax in axes:
        ax.set(xscale='log', xlabel='Query budget ρ', xticks=[.01, .025, .05, .1, .2])
        ax.set_xticklabels(['.01', '.025', '.05', '.1', '.2'])
        ax.xaxis.set_minor_formatter(NullFormatter())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.50, .005), ncol=3, handlelength=2.5, columnspacing=2)
    save(fig, axes, 'fig4_budget', ['budget_profiles.csv', 'test_aggregate_metrics.csv', 'test_geometry_summary.csv'],
         {'profiles_retained': 120, 'settings_per_strategy': 12, 'budgets': 5,
          'uncertainty_exemption': 'Descriptive equal-setting dataset means. All 12 individual setting profiles show between-setting dispersion; no confidence-band inference or test-based budget selection.',
          'disabled_displacement': 'NaN only for inactive policies; all score profiles retained and inactivity documented in Source Data.'})

def transfer():
    fig, axes = plt.subplots(1, 3, figsize=(183/25.4, 88/25.4))
    fig.subplots_adjust(left=.20, right=.975, top=.85, bottom=.19, wspace=.30)
    for i, (family, baseline, title) in enumerate([
        ('primary', 'frozen', 'Frozen source'), ('primary', 'nlms_expanded', 'Fixed NLMS'),
        ('independently_tuned_history', 'history_clip', 'Fixed historical clipping')]):
        panel(axes[i], 'abc'[i], title)
        forest(axes[i], d.comparisons('withheld', family, baseline), d.NEWCELLS, i == 0)
    fig.text(.20, .045, 'Negative: lower MAE. Filled: Holm-adjusted p < 0.05.', fontsize=6.5)
    save(fig, axes, 'fig5_entity_transfer', ['withheld_paired_comparisons.csv'],
         {'settings_per_panel': 8, 'seeds_per_setting': 5, 'uncertainty': 'pointwise paired 95% seed/calendar-block intervals; 7-day blocks',
          'adverse_accuracy_outcomes_retained': True})

if __name__ == '__main__':
    selected = sys.argv[1:]
    for name, method in [('workflow', workflow), ('controls', controls), ('budget', budget), ('accuracy', accuracy), ('transfer', transfer)]:
        if not selected or name in selected: method()
    result = subprocess.run([sys.executable, str(HERE / 'figure_helpers/validate_figure.py'), str(Path(__file__)),
                             '--backend', 'python', '--json'], capture_output=True, text=True)
    (Q / 'source_validation.json').write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode: raise RuntimeError(result.stdout + result.stderr)
