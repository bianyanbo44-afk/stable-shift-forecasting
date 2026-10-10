"""Describe current-query displacement against a common event budget.

This post-hoc summary uses the complete locked update records. It does not
change selections, trajectories, budgets, bootstrap families or primary scores.
NLMS records store a zero internal cap because NLMS is unconstrained; every
comparison therefore joins the QNC cap from the matching feedback event.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
A = HERE / 'analysis'
T = HERE / 'tables'
KEYS = ['dataset', 'horizon', 'base', 'seed', 'update_number']
METHODS = {'query_expanded': 'QNC', 'nlms_matched_new': 'Matched NLMS',
           'history_matched_new': 'Matched history'}
PHASES = {'test': 'Original benchmark', 'independent': 'EIA re-evaluation',
          'withheld': 'Unused entities'}


def main():
    rows = []
    hashes = {}
    for phase in PHASES:
        path = A / f'{phase}_update_geometry.csv'
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        geometry = pd.read_csv(path)
        reference = geometry[geometry.method == 'query_expanded'][KEYS + ['query_cap']].rename(columns={'query_cap': 'common_cap'})
        assert not reference.duplicated(KEYS).any()
        for method in METHODS:
            part = geometry[geometry.method == method].merge(reference, on=KEYS, how='left', validate='one_to_one')
            assert len(part) == len(reference)
            assert part.common_cap.notna().all() and (part.common_cap > 0).all()
            ratio = part.query_rms / part.common_cap
            assert np.isfinite(ratio).all()
            exceed = part.query_rms > part.common_cap + 2e-6 * np.maximum(1, part.common_cap)
            if method == 'query_expanded':
                assert not exceed.any()
            rows.append({'phase': phase, 'method': method, 'updates': len(part),
                         'exceedances': int(exceed.sum()), 'exceedance_fraction': float(exceed.mean()),
                         'ratio_p99': float(ratio.quantile(.99)), 'ratio_max': float(ratio.max())})
    frame = pd.DataFrame(rows)
    frame.to_csv(A / 'control_boundary_summary.csv', index=False)
    lines = []
    for row in frame.itertuples():
        lines.append(f'{PHASES[row.phase]} & {METHODS[row.method]} & {row.updates:,} & {100*row.exceedance_fraction:.2f} & {row.ratio_p99:.2f} & {row.ratio_max:.2f}\\\\')
    table = '\\begin{tabular}{llrrrr}\n\\toprule\nEvaluation & Update rule & Events & Over budget (\\%) & P99 ratio & Max ratio\\\\\n\\midrule\n' + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n'
    (T / 'control_boundary_summary.tex').write_text(table, encoding='utf-8')
    audit = {'analysis_status': 'post-hoc descriptive summary of locked trajectories',
             'input_sha256': hashes, 'join_keys': KEYS, 'common_budget': 'QNC query_cap from the same event',
             'exceedance_tolerance': '2e-6 * max(1, common_cap) in standardized RMS units',
             'active_events_only': True, 'all_events_retained': True,
             'new_model_or_budget_selection': False, 'rows': rows}
    (A / 'control_boundary_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(frame.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
