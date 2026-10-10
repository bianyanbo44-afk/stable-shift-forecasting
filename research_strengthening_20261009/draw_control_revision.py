"""Put the current-query control evidence before accuracy and cost panels.

Python structural adaptation of the existing four-panel figure. All update
events, twelve paired comparisons and eleven active timing pairs are retained.
The panel roles are control capability, compute cost, normalization control,
and query-independent clipping control. No statistic is reselected.
"""
import json
import subprocess
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import draw_strengthening as d
from audit_panel_alignment import require_matplotlib_panel_alignment

HERE = Path(__file__).resolve().parent
d.F = Path(os.environ.get('QNC_FIGURE_DIR',str(HERE / 'control_figures')))
d.Q = Path(os.environ.get('QNC_FIGURE_QA_DIR',str(HERE / 'control_figure_quality')))
d.F.mkdir(exist_ok=True)
d.Q.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'ps.fonttype': 42})


def export(fig, axes):
    name = 'fig2_control_first'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=list(axes.ravel()), panel_ids=list('abcd'),
        json_out=d.Q / f'{name}_alignment.json', overlay_svg=d.Q / f'{name}_alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(d.F / f'{name}.pdf')
    fig.savefig(d.F / f'{name}.svg')
    fig.savefig(d.F / f'{name}.png', dpi=600)
    fig.savefig(d.F / f'{name}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    for script, args in [('audit_pdf_text.py', ['--min-pt', '5', '--json']),
                         ('audit_figure_collisions.py', ['--json-out', str(d.Q / f'{name}_collision.json'),
                           '--overlay-pdf', str(d.Q / f'{name}_collision.pdf'), '--json'])]:
        result = subprocess.run([sys.executable, str(d.HELPER / script), str(d.F / f'{name}.pdf'), *args],
                                 capture_output=True, text=True)
        (d.Q / f'{name}_{script}.txt').write_text(result.stdout, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    d.MANIFEST[name] = {'sources': ['test_update_geometry.csv', 'runtime_summary.csv',
                                  'runtime_repetitions.csv', 'test_paired_comparisons.csv'],
                       'width_mm': 183, 'raster_dpi': 600, 'all_predeclared_settings_retained': True}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.2047244, 5.7874016))
    fig.subplots_adjust(left=.205, right=.975, top=.91, bottom=.10, wspace=.72, hspace=.66)
    control, runtime_ax, nlms, history = axes.ravel()
    d.panel(control, 'a', 'Current-query displacement')
    geometry = pd.read_csv(d.A / 'test_update_geometry.csv')
    keys = ['dataset', 'horizon', 'base', 'seed', 'update_number']
    reference = geometry[geometry.method == 'query_expanded'][keys + ['query_cap']].rename(columns={'query_cap': 'reference_cap'})
    audits = []
    for method, color, label in [('query_expanded', d.BLUE, d.DISPLAY_METHOD),
                                ('history_matched_new', d.TEAL, 'Historical clipping'),
                                ('nlms_matched_new', d.ORANGE, 'NLMS')]:
        part = geometry[geometry.method == method].merge(reference, on=keys, validate='one_to_one')
        assert len(part) == len(reference) and (part.reference_cap > 0).all()
        values = np.sort((part.query_rms / part.reference_cap).to_numpy())
        assert np.isfinite(values).all() and (values >= 0).all()
        control.step(values, np.arange(1, len(values) + 1) / len(values), where='post',
                     color=color, lw=1.4, label=label, zorder=4 if method == 'query_expanded' else 2)
        audits.append({'method': method, 'updates': len(values),
                       'ratio_max': float(values.max()), 'ratio_p99': float(np.quantile(values, .99))})
    control.axvline(1, color=d.GRAY, ls='--', lw=.8, zorder=1)
    limit = max(item['ratio_max'] for item in audits) * 1.03
    control.set_xscale('symlog', linthresh=1)
    control.set(xlim=(0, limit), ylim=(0, 1.03), xlabel='Current RMS change / query budget', ylabel='Fraction of active updates')
    control.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
    control.xaxis.set_minor_formatter(NullFormatter())
    control.legend(loc='lower right', fontsize=7)
    control.grid(axis='y', color=d.LIGHT, lw=.6)
    control.set_axisbelow(True)
    (d.Q / 'geometry_display.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')

    d.panel(runtime_ax, 'b', 'Unbatched calibration and scoring')
    timings = pd.read_csv(d.A / 'runtime_summary.csv').pivot(index=['dataset', 'horizon', 'base'], columns='method', values='ms_per_origin')
    repetitions = pd.read_csv(d.A / 'runtime_repetitions.csv')
    active = repetitions.groupby(['dataset', 'horizon', 'base', 'method']).updates.max().unstack()
    selected = timings[(active.query_expanded > 0) & (active.cosa_expanded > 0)]
    assert len(selected) == 11
    x, y = selected.cosa_expanded, selected.query_expanded
    for dataset in d.DS:
        mask = selected.index.get_level_values('dataset') == dataset
        runtime_ax.scatter(x[mask], y[mask], s=20, color=d.COLORS[dataset], edgecolor='white',
                           linewidth=.3, zorder=3, label=d.DS[dataset])
    lower, upper = min(x.min(), y.min()) * .65, max(x.max(), y.max()) * 1.4
    runtime_ax.plot([lower, upper], [lower, upper], color=d.GRAY, lw=.8, ls='--')
    runtime_ax.set(xscale='log', yscale='log', xlim=(lower, upper), ylim=(lower, upper),
                   xlabel='Controlled COSA (ms/origin)', ylabel=f'{d.DISPLAY_METHOD} (ms/origin)')
    for axis in (runtime_ax.xaxis, runtime_ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        axis.set_minor_formatter(NullFormatter())
    runtime_ax.legend(loc='upper left', fontsize=7, handlelength=.8)
    d.panel(nlms, 'c', 'Matched-step NLMS')
    d.forest(nlms, d.comparisons('test', 'matched_mechanism', 'nlms_matched_new'), d.CELLS)
    d.panel(history, 'd', 'Matched historical clipping')
    d.forest(history, d.comparisons('test', 'matched_mechanism', 'history_matched_new'), d.CELLS, False)
    export(fig, axes)
    validation = subprocess.run([sys.executable, str(d.HELPER / 'validate_figure.py'), str(Path(__file__)),
                                 '--backend', 'python', '--json'], capture_output=True, text=True)
    (d.Q / 'source_validation.json').write_text(validation.stdout, encoding='utf-8')
    if validation.returncode:
        raise RuntimeError(validation.stdout + validation.stderr)
    (d.Q / 'figure_manifest.json').write_text(json.dumps(d.MANIFEST, indent=2), encoding='utf-8')
    print('CONTROL-FIRST FIGURE EXPORTED AND AUDITED', flush=True)


if __name__ == '__main__':
    main()
