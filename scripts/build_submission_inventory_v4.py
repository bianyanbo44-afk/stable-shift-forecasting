"""Inventory completed result units for the Neurocomputing submission.

The report keeps result roots separate when their protocol signatures differ.
It is intentionally descriptive: no missing run is promoted to a completed
benchmark cell and no metric is synthesized from a partial seed set.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_root(path: Path) -> dict:
    protocol_path = path / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
    cfg = protocol.get("config", {})
    datasets = cfg.get("datasets", {})
    seeds = cfg.get("seeds", [11, 22, 33, 44, 55])
    horizons = cfg.get("horizons", [24, 48, 168])
    methods = cfg.get("methods", [])
    cells = []
    for dataset in datasets:
        for horizon in horizons:
            folder = path / dataset / f"h{horizon}"
            if not folder.exists():
                continue
            method_counts = {}
            for method in methods:
                count = sum((folder / f"seed{seed}" / method / "completed.json").exists() for seed in seeds)
                if count:
                    method_counts[method] = count
            required = methods
            complete = all(method_counts.get(method, 0) == len(seeds) for method in required)
            cells.append({
                "dataset": dataset,
                "horizon": horizon,
                "method_seed_counts": method_counts,
                "complete_all_methods": complete,
                "complete_core_methods": all(
                    method_counts.get(method, 0) == len(seeds)
                    for method in ["SeasonalNaive", "DLinear", "PatchTST", "iTransformer", "TiDE", "StableShift"]
                    if method in methods
                ),
            })
    return {
        "root": str(path),
        "protocol_signature": protocol.get("signature"),
        "source_digest": protocol.get("source_digest"),
        "device": protocol.get("gpu", cfg.get("device")),
        "configuration": {
            "context_length": cfg.get("context_length"),
            "horizons": horizons,
            "seeds": seeds,
            "methods": methods,
            "epochs": cfg.get("epochs"),
            "stable_model": cfg.get("stable_model"),
        },
        "cells": cells,
    }


def main() -> None:
    roots = [
        ROOT / "results" / "benchmark_v3_formal",
        ROOT / "cloud_returned_20260909" / "results" / "cloud_ecl_v3",
        ROOT / "cloud_returned_20260909" / "results" / "cloud_ecl_h168_v3",
        ROOT / "cloud_returned_20260909" / "results" / "cloud_ausgrid_v3",
        ROOT / "cloud_returned_20260909" / "results" / "cloud_pjm_v3",
        ROOT / "cloud_returned_20260909" / "results" / "benchmark_v3_transfer",
        ROOT / "results" / "aggregate",
    ]
    report = {"generated_by": str(Path(__file__).relative_to(ROOT)), "roots": []}
    for root in roots:
        if (root / "protocol.json").exists():
            report["roots"].append(inspect_root(root))
    report["legacy_primary_aggregate"] = {
        "path": str(ROOT / "results" / "aggregate"),
        "metadata_sha256": digest(ROOT / "results" / "aggregate" / "aggregate_metadata.json"),
        "claim_audit_sha256": digest(ROOT / "results" / "aggregate" / "claim_audit_final.json"),
        "note": "Historical only. Excluded from the revised manuscript: the v3 observed-label benchmark supersedes the compact pilot protocol.",
    }
    destination = ROOT / "submission" / "result_inventory_v4.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination), "roots": len(report["roots"])}, indent=2))


if __name__ == "__main__":
    main()
