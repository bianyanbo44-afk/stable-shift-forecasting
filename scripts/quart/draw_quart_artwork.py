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
import csv
import importlib.util
import importlib.metadata
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True

HELPER_FILES = ('audit_panel_alignment.py', 'audit_pdf_text.py',
                'audit_figure_collisions.py', 'validate_figure.py')
REQUIRED_INPUTS = {
    'test_update_geometry.csv': {'dataset', 'horizon', 'base', 'seed', 'update_number',
                                 'method', 'query_cap', 'query_rms'},
    'runtime_summary.csv': {'dataset', 'horizon', 'base', 'method', 'ms_per_origin'},
    'runtime_repetitions.csv': {'dataset', 'horizon', 'base', 'method', 'updates'},
    'test_paired_comparisons.csv': {'family', 'baseline', 'metric', 'block_days',
                                  'dataset', 'horizon', 'base', 'relative_difference',
                                  'relative_ci_low', 'relative_ci_high', 'holm_p'},
    'withheld_paired_comparisons.csv': {'family', 'baseline', 'metric', 'block_days',
                                      'dataset', 'horizon', 'base', 'relative_difference',
                                      'relative_ci_low', 'relative_ci_high', 'holm_p'},
    'budget_profiles.csv': {'strategy', 'dataset', 'horizon', 'base', 'rho',
                            'relative_mae', 'query_rms_p99'},
    'test_aggregate_metrics.csv': set(),
    'test_geometry_summary.csv': set(),
}

def resolve_artwork_paths(script_dir, working_dir=None, environment=None):
    """Honor explicit overrides, then use nearby reproducibility layouts."""
    script_dir = Path(script_dir).resolve()
    working_dir = Path.cwd() if working_dir is None else Path(working_dir).resolve()
    environment = os.environ if environment is None else environment
    explicit = environment.get('QUART_SUBMISSION_DIR') or environment.get('QURB_SUBMISSION_DIR')
    if explicit:
        output = Path(explicit).expanduser()
        output = (working_dir / output).resolve() if not output.is_absolute() else output.resolve()
        if output.name == 'source_data':
            output = output.parent
        output_mode = 'QUART_SUBMISSION_DIR'
    else:
        conventional = script_dir.parent / 'submission_neurocomputing_nature_editor_20261011'
        candidates = [(conventional, 'repository-relative'),
                      (script_dir, 'script-relative'),
                      (script_dir.parent, 'parent-relative'),
                      (working_dir, 'working-directory-relative')]
        output, output_mode = next(((path, mode) for path, mode in candidates
                                    if (path / 'source_data').is_dir()),
                                   (conventional, 'repository-relative-inputs-missing'))
    if output_mode.startswith('repository-relative') or output_mode == 'QUART_SUBMISSION_DIR' or (output / 'source').is_dir():
        figure_dir = output / 'source/figures'
    else:
        figure_dir = output / 'figures'
    explicit_helpers = environment.get('NATURE_FIGURE_HELPERS')
    if explicit_helpers:
        helpers = Path(explicit_helpers).expanduser()
        helpers = (working_dir / helpers).resolve() if not helpers.is_absolute() else helpers.resolve()
        helper_mode = 'NATURE_FIGURE_HELPERS'
    else:
        candidates = [(script_dir / 'portable_figure_helpers', 'bundled-relative'),
                      (script_dir / 'helpers', 'script-relative-helpers'),
                      (script_dir / 'figure_helpers', 'script-relative-figure-helpers'),
                      (output / 'portable_figure_helpers', 'output-relative'),
                      (output / 'helpers', 'output-relative-helpers'),
                      (script_dir.parent / 'portable_figure_helpers', 'parent-relative'),
                      (script_dir.parent / 'helpers', 'parent-relative-helpers'),
                      (Path.home() / '.codex/skills/nature-figure/scripts', 'installed-skill-fallback')]
        helpers, helper_mode = next(((path, mode) for path, mode in candidates
                                     if all((path / name).is_file() for name in HELPER_FILES)),
                                    (script_dir / 'portable_figure_helpers', 'helpers-missing'))
    return {'output': output, 'data': output / 'source_data', 'figures': figure_dir,
            'quality': output / 'quality/artwork', 'helpers': helpers,
            'output_mode': output_mode, 'helper_mode': helper_mode}

def inspect_artwork_inputs(paths):
    """Read-only file/schema/dependency check. This function never draws."""
    report = {'paths': {key: str(value) for key, value in paths.items()},
              'inputs': {}, 'helpers': {}, 'python_dependencies': {}, 'ready': True}
    for name, required in REQUIRED_INPUTS.items():
        path = paths['data'] / name
        entry = {'exists': path.is_file()}
        if path.is_file():
            with path.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                columns = set(reader.fieldnames or [])
                entry.update({'columns': sorted(columns), 'missing_columns': sorted(required-columns),
                              'has_data_row': next(reader, None) is not None,
                              'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            entry['complete'] = bool(columns) and not entry['missing_columns'] and entry['has_data_row']
        else:
            entry['complete'] = False
        report['inputs'][name] = entry
        report['ready'] = report['ready'] and entry['complete']
    for name in HELPER_FILES:
        path = paths['helpers'] / name
        report['helpers'][name] = {'exists': path.is_file(),
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}
        report['ready'] = report['ready'] and path.is_file()
    for module, package in [('numpy', 'numpy'), ('pandas', 'pandas'), ('matplotlib', 'matplotlib'),
                            ('PIL', 'Pillow'), ('fitz', 'PyMuPDF')]:
        available = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        report['python_dependencies'][package] = {'available': available, 'version': version}
        report['ready'] = report['ready'] and available
    return report

PATHS = resolve_artwork_paths(HERE)
OUT, F, Q, HELPERS = (PATHS[key] for key in ('output', 'figures', 'quality', 'helpers'))
if __name__ == '__main__' and any(argument in ('--help', '-h') for argument in sys.argv[1:]):
    print('Usage: python draw_quart_artwork.py [workflow|controls|budget|accuracy|transfer]')
    print('       python draw_quart_artwork.py --check-inputs')
    print('Overrides: QUART_SUBMISSION_DIR (package root), NATURE_FIGURE_HELPERS (QA helper directory).')
    sys.exit(0)
if __name__ == '__main__':
    input_check = inspect_artwork_inputs(PATHS)
    if '--check-inputs' in sys.argv[1:] or not input_check['ready']:
        print(json.dumps(input_check, indent=2))
        sys.exit(0 if input_check['ready'] else 2)
    unknown = set(sys.argv[1:]) - {'workflow', 'controls', 'budget', 'accuracy', 'transfer'}
    if unknown:
        raise SystemExit(f'Unknown figure selection: {sorted(unknown)}. Use --help.')
Q.mkdir(parents=True, exist_ok=True)
F.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(Q / '.matplotlib'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.transforms import ScaledTranslation
import fitz

sys.path.insert(0, str(HELPERS))
from audit_panel_alignment import require_matplotlib_panel_alignment

# The recorded scientific mappings are retained from the original workflow.
d = SimpleNamespace(A=OUT / 'source_data', DS={'eia930': 'EIA', 'ecl': 'ECL', 'ausgrid': 'Ausgrid'})
d.CELLS = [(ds, horizon, base) for ds in d.DS for horizon in [24, 168] for base in ['DLinear', 'StableShift']]
d.NEWCELLS = [cell for cell in d.CELLS if cell[0] != 'eia930']
d.LABELS = lambda cells: [f'{d.DS[ds]} · {base} · {horizon} h' for ds, horizon, base in cells]
NUMERICAL_ROWS = []
SOURCE_SNAPSHOT = Q / 'source_data_before.json'
if not SOURCE_SNAPSHOT.is_file():
    snapshot = {str(path.relative_to(d.A)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(d.A.rglob('*')) if path.is_file()}
    SOURCE_SNAPSHOT.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')

def recorded_comparisons(phase, family, baseline):
    path = d.A / f'{phase}_paired_comparisons.csv'
    frame = pd.read_csv(path)
    selected = frame[(frame.family == family) & (frame.baseline == baseline) &
                     (frame.metric == 'mae') & (frame.block_days == 7)].copy()
    selected['_source'] = path.name
    return selected

def recorded_forest(ax, frame, cells, show_labels=True, color='#176B8F'):
    n = len(cells)
    assert len(frame) == n, 'Every predeclared setting must be represented once.'
    for i, (ds, horizon, base) in enumerate(cells):
        rows = frame[(frame.dataset == ds) & (frame.horizon == horizon) & (frame.base == base)]
        assert len(rows) == 1, (ds, horizon, base, len(rows))
        row = rows.iloc[0]
        point, low, high = 100 * np.array([row.relative_difference, row.relative_ci_low, row.relative_ci_high])
        assert np.isfinite([point, low, high]).all() and low <= high
        y = n - 1 - i
        interval, = ax.plot([low, high], [y, y], color=color, lw=1.15, zorder=2)
        ax.plot([low, low], [y-.08, y+.08], color=color, lw=.75)
        ax.plot([high, high], [y-.08, y+.08], color=color, lw=.75)
        filled = bool(row.holm_p < .05)
        marker, = ax.plot(point, y, 'o', ms=4, mec=color, mfc=color if filled else 'white', mew=1.05, zorder=3)
        assert np.array_equal(interval.get_xdata(), [low, high])
        assert np.array_equal(marker.get_xdata(), [point])
        NUMERICAL_ROWS.append({'source': row['_source'], 'family': row.family, 'baseline': row.baseline,
                               'dataset': ds, 'horizon': horizon, 'base': base,
                               'point_percent': float(point), 'ci_low_percent': float(low),
                               'ci_high_percent': float(high), 'holm_p': float(row.holm_p), 'filled': filled})
    ax.set_ylim(-.65, n-.35)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(d.LABELS(cells)[::-1] if show_labels else [''] * n)
    ax.tick_params(axis='y', length=0, pad=5)
    if n == 12:
        ax.axhspan(3.5, 7.5, color='#F5F7F8', zorder=-3)
    ax.axvline(0, color=GRAY, lw=.75)
    ax.set_axisbelow(True)
    low = min(-.25, 100 * frame.relative_ci_low.min())
    high = max(.25, 100 * frame.relative_ci_high.max())
    padding = (high-low) * .07
    ax.set_xlim(low-padding, high+padding)

d.comparisons = recorded_comparisons
d.forest = recorded_forest

BLUE, ORANGE, PURPLE = '#176B8F', '#C46D3F', '#8064A2'
GRAY, DARK, LIGHT = '#68727B', '#20252B', '#B7BEC4'
DATASETS = {'eia930': (BLUE, 'o', '-'), 'ecl': (ORANGE, 's', '--'), 'ausgrid': (PURPLE, '^', ':')}
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8.5, 'axes.titlesize': 8.5,
    'axes.labelsize': 8.5, 'xtick.labelsize': 8.2, 'ytick.labelsize': 8.2,
    'legend.fontsize': 8.2, 'text.color': DARK, 'axes.labelcolor': DARK,
    'axes.edgecolor': DARK, 'xtick.color': DARK, 'ytick.color': DARK,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
    'legend.frameon': False, 'savefig.facecolor': 'white', 'axes.grid': False,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'path.simplify': False, 'agg.path.chunksize': 0, 'savefig.bbox': None,
})
MANIFEST = json.loads((Q / 'figure_manifest.json').read_text(encoding='utf-8')) if (Q / 'figure_manifest.json').is_file() else {}

def panel(ax, letter, title):
    # Fixed point offsets keep panel anchors aligned at any axes height.
    transform = ax.transAxes + ScaledTranslation(-9/72, 10/72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, letter, transform=transform, fontsize=9, weight='bold', va='bottom')
    ax.set_title(title, loc='left', pad=12, color=DARK, fontsize=8.5)

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
        ('audit_pdf_text.py', ['--min-pt', '8', '--json']),
        ('audit_figure_collisions.py', ['--json-out', str(Q / f'{name}_collision.json'),
                                      '--overlay-pdf', str(Q / f'{name}_collision.pdf'), '--json'])]:
        result = subprocess.run([sys.executable, str(HELPERS / script),
                                 str(F / f'{name}.pdf'), *args], capture_output=True, text=True, encoding='utf-8')
        (Q / f'{name}_{script}.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    target_width_mm = 160.72 if name == 'fig2_control_first' else 164.0
    with fitz.open(F / f'{name}.pdf') as document:
        page = document[0]
        width_mm = page.rect.width * 25.4 / 72
        spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block
                 for line in block['lines'] for span in line['spans'] if span['text'].strip()]
        scale = target_width_mm / width_mm
        minimum_embedded_pt = min(span['size'] * scale for span in spans)
        assert abs(width_mm-target_width_mm) < .01
        assert minimum_embedded_pt >= 8 - .01
        page.get_pixmap(matrix=fitz.Matrix(180/72, 180/72)).save(Q / f'{name}_preview.png')
    def finite_list(values):
        return [float(value) if np.isfinite(value) else None for value in np.asarray(values).ravel()]
    drawn = []
    for i, ax in enumerate(list(np.asarray(axes).ravel())):
        drawn.append({'panel': list('abcd')[i], 'xlim': list(ax.get_xlim()), 'ylim': list(ax.get_ylim()),
                      'lines': [{'x': finite_list(line.get_xdata()), 'y': finite_list(line.get_ydata()),
                                 'label': line.get_label(), 'marker': line.get_marker(),
                                 'markerfacecolor': str(line.get_markerfacecolor())} for line in ax.lines],
                      'scatter_offsets': [finite_list(collection.get_offsets()) for collection in ax.collections]})
    (Q / f'{name}_numerical_preservation.json').write_text(
        json.dumps({'recorded_comparison_rows': list(NUMERICAL_ROWS), 'drawn_panels': drawn,
                    'no_statistical_reselection': True, 'no_path_simplification': True}, indent=2) + '\n', encoding='utf-8')
    NUMERICAL_ROWS.clear()
    MANIFEST[name] = {
        'width_mm': width_mm, 'height_mm': fig.get_size_inches()[1] * 25.4,
        'intended_main_width_mm': target_width_mm, 'embed_scale': scale,
        'minimum_embedded_font_pt': minimum_embedded_pt,
        'sources': sources, 'source_sha256': {p: hashlib.sha256((d.A / p).read_bytes()).hexdigest() for p in sources},
        'raster_dpi': 600, 'all_predeclared_settings_retained': True,
        'new_statistical_selection': False, 'alignment': alignment['verdict'], **details}
    (Q / 'figure_manifest.json').write_text(json.dumps(MANIFEST, indent=2) + '\n', encoding='utf-8')
    print('PASS:', name, flush=True)

def forest(ax, frame, cells, show_labels=True):
    d.forest(ax, frame, cells, show_labels, color=BLUE)
    ax.grid(False)
    ax.set_yticklabels(d.LABELS(cells)[::-1] if show_labels else [''] * len(cells), fontsize=8.2)
    ax.tick_params(axis='y', pad=4)
    ax.set_xlabel('QUART MAE\ndifference (%)', fontsize=8.5, linespacing=1.15)

def workflow():
    fig, axes = plt.subplots(1, 3, figsize=(164/25.4, 75/25.4))
    fig.subplots_adjust(left=.035, right=.98, bottom=.14, top=.80, wspace=.30)
    for ax, letter, title in zip(axes, 'abc', ['Frozen forecast', 'Delayed feedback', 'Query constraint']):
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.set_aspect('equal', adjustable='box')
        ax.axis('off'); panel(ax, letter, title)
    def box(ax, x, y, w, h, text, accent=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.012,rounding_size=.015',
                                   facecolor='#F5F7F8', edgecolor=accent, linewidth=.8))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=8.2, color=DARK, linespacing=1.15)
    def arrow(ax, start, end, color=GRAY, width=.8):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=7, color=color, linewidth=width))
    ax = axes[0]
    box(ax, .05, .73, .90, .20, 'Observed load + calendar\nPast 336 h', GRAY)
    arrow(ax, (.50, .71), (.50, .55))
    box(ax, .12, .35, .76, .20, 'Frozen source model')
    arrow(ax, (.50, .33), (.50, .25), BLUE)
    box(ax, .05, .04, .90, .19, 'Source forecast\nNext 24 or 168 h')
    ax.text(.5, -.11, 'Daily issuance', ha='center', fontsize=8.2)
    ax = axes[1]
    box(ax, .02, .64, .43, .34, 'Earlier\nforecast\nObserved load', GRAY)
    box(ax, .55, .64, .43, .34, 'Current\nforecast\nObserved\nhistory')
    arrow(ax, (.235, .62), (.34, .46), GRAY)
    arrow(ax, (.765, .62), (.66, .46), BLUE)
    ax.text(-.015, .51, 'Residual', fontsize=8.2)
    ax.text(.75, .51, 'Query', fontsize=8.2)
    box(ax, .09, .23, .82, .23, 'Normalised proposal\n+ query-dependent step')
    arrow(ax, (.50, .21), (.50, .075), BLUE)
    ax.text(.50, .025, 'Update the output operator', ha='center', fontsize=8.5)
    ax.text(.50, -.11, 'Each matured window used once', ha='center', fontsize=8.2)
    ax = axes[2]
    centre, end = np.array([.34, .39]), np.array([.88, .83])
    edge = centre + .285 * (end-centre) / np.linalg.norm(end-centre)
    ax.add_patch(Circle(centre, .285, facecolor='#F2F6F8', edgecolor=BLUE, linewidth=.85))
    arrow(ax, centre, end, ORANGE, 1.2); arrow(ax, centre, edge, BLUE, 1.6)
    ax.plot(*centre, 'o', color=DARK, markersize=3)
    ax.text(.31, .015, 'Before update', ha='center', fontsize=8.2)
    ax.text(.82, .91, 'Proposal', ha='center', fontsize=8.2)
    ax.text(.27, .80, 'Allowed\nRMS change', ha='center', fontsize=8.2, linespacing=1.25)
    ax.text(.82, .40, 'QUART\nrevision', ha='center', fontsize=8.2, linespacing=1.25)
    ax.text(.50, -.11, 'Issue the corrected forecast', ha='center', fontsize=8.2)
    save(fig, axes, 'fig1_method', [], {'archetype': 'schematic-led composite', 'empirical_data': False})

def controls():
    fig, axes = plt.subplots(2, 2, figsize=(160.72/25.4, 166/25.4))
    fig.subplots_adjust(left=.282, right=.985, top=.925, bottom=.095, wspace=.45, hspace=.56)
    control, timing, nlms, history = axes.ravel()
    panel(control, 'a', 'Current-query displacement')
    geometry = pd.read_csv(d.A / 'test_update_geometry.csv')
    keys = ['dataset', 'horizon', 'base', 'seed', 'update_number']
    reference = geometry[geometry.method == 'query_expanded'][keys + ['query_cap']].rename(columns={'query_cap': 'reference_cap'})
    audit = []
    for method, color, label, style in [
        ('query_expanded', BLUE, 'QUART', '-'),
        ('history_matched_new', GRAY, 'Matched historical\nclipping', '--'),
        ('nlms_matched_new', ORANGE, 'Matched NLMS', ':')]:
        part = geometry[geometry.method == method].merge(reference, on=keys, validate='one_to_one')
        assert len(part) == len(reference) and (part.reference_cap > 0).all()
        values = np.sort((part.query_rms / part.reference_cap).to_numpy())
        assert np.isfinite(values).all() and (values >= 0).all()
        empirical_y = np.arange(1, len(values)+1)/len(values)
        curve, = control.step(values, empirical_y, where='post',
                     color=color, linestyle=style, linewidth=1.25,
                     label=label, zorder=4 if method == 'query_expanded' else 2)
        assert np.array_equal(curve.get_xdata(), values) and np.array_equal(curve.get_ydata(), empirical_y)
        audit.append({'method': method, 'updates': len(values), 'ratio_max': float(values.max()),
                      'complete_values_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
                      'ratio_p99': float(np.quantile(values, .99))})
    control.axvline(1, color=GRAY, linestyle='--', linewidth=.6, zorder=1)
    control.set_xscale('symlog', linthresh=1)
    control.set(xlim=(0, max(r['ratio_max'] for r in audit)*1.03), ylim=(0, 1.03),
                xlabel='RMS change / query budget', ylabel='Fraction of active updates')
    control.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    control.xaxis.set_minor_formatter(NullFormatter())
    control.legend(loc='lower right', handlelength=1.8, labelspacing=.5, borderpad=.2)
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
               xlabel='Controlled COSA (ms/origin)', ylabel='QUART (ms/origin)')
    for axis in (timing.xaxis, timing.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}')); axis.set_minor_formatter(NullFormatter())
    timing.legend(loc='upper left', handlelength=1, labelspacing=.4)
    timing.text(.74, .34, f'{(x/y).median():.1f}× median\nControlled COSA\n/ QUART', transform=timing.transAxes,
                fontsize=8.2, ha='center', va='center', color=DARK)
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
    fig, axes = plt.subplots(1, 3, figsize=(164/25.4, 112/25.4))
    fig.subplots_adjust(left=.282, right=.985, top=.855, bottom=.205, wspace=.33)
    for i, (baseline, title) in enumerate([('frozen', 'Frozen source'), ('cosa_expanded', 'Controlled COSA'), ('nlms_expanded', 'Tuned NLMS')]):
        panel(axes[i], 'abc'[i], title)
        forest(axes[i], d.comparisons('test', 'primary', baseline), d.CELLS, i == 0)
    fig.text(.282, .045, 'Negative: lower MAE. Filled: Holm-adjusted p < 0.05.', fontsize=8.2)
    save(fig, axes, 'fig2_expanded_accuracy', ['test_paired_comparisons.csv'],
         {'settings_per_panel': 12, 'seeds_per_setting': 5, 'uncertainty': 'pointwise paired 95% seed/calendar-block intervals; 7-day blocks'})

def budget():
    data = pd.read_csv(d.A / 'budget_profiles.csv')
    assert len(data) == 120 and set(data.rho) == {.01, .025, .05, .1, .2}
    assert data.relative_mae.between(-12, 2).all(), 'Shared display limits must contain every score.'
    fig, axes = plt.subplots(1, 3, figsize=(164/25.4, 93/25.4))
    fig.subplots_adjust(left=.105, right=.985, top=.84, bottom=.245, wspace=.43)
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
    fig, axes = plt.subplots(1, 3, figsize=(164/25.4, 100/25.4))
    fig.subplots_adjust(left=.282, right=.985, top=.855, bottom=.235, wspace=.33)
    for i, (family, baseline, title) in enumerate([
        ('primary', 'frozen', 'Frozen source'), ('primary', 'nlms_expanded', 'Fixed NLMS'),
        ('independently_tuned_history', 'history_clip', 'Fixed historical clipping')]):
        panel(axes[i], 'abc'[i], title)
        forest(axes[i], d.comparisons('withheld', family, baseline), d.NEWCELLS, i == 0)
    fig.text(.282, .047, 'Negative: lower MAE. Filled: Holm-adjusted p < 0.05.', fontsize=8.2)
    save(fig, axes, 'fig5_entity_transfer', ['withheld_paired_comparisons.csv'],
         {'settings_per_panel': 8, 'seeds_per_setting': 5, 'uncertainty': 'pointwise paired 95% seed/calendar-block intervals; 7-day blocks',
          'adverse_accuracy_outcomes_retained': True})

if __name__ == '__main__':
    result = subprocess.run([sys.executable, str(HELPERS / 'validate_figure.py'), str(Path(__file__)),
                             '--backend', 'python', '--json'], capture_output=True, text=True, encoding='utf-8')
    (Q / 'source_validation.json').write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode: raise RuntimeError(result.stdout + result.stderr)
    selected = sys.argv[1:]
    for name, method in [('workflow', workflow), ('controls', controls), ('budget', budget), ('accuracy', accuracy), ('transfer', transfer)]:
        if not selected or name in selected: method()
    before = json.loads(SOURCE_SNAPSHOT.read_text(encoding='utf-8'))
    after = {str(path.relative_to(d.A)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(d.A.rglob('*')) if path.is_file()}
    assert before == after, 'Source Data must stay byte-identical.'
    (Q / 'source_data_after.json').write_text(json.dumps(after, indent=2) + '\n', encoding='utf-8')
