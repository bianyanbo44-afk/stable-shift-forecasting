"""Download a declared public data source and record its digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_URLS = {
    "ecl": "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
    "pjm": "https://www.kaggle.com/api/v1/datasets/download/robikscube/hourly-energy-consumption",
    "ausgrid": "https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DEFAULT_URLS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    url = args.url or DEFAULT_URLS[args.dataset]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "stable-shift-forecasting/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, args.output.open("wb") as destination:
        shutil.copyfileobj(response, destination)
    record = {"dataset": args.dataset, "url": url, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": sha256(args.output), "path": str(args.output)}
    manifest = args.manifest or args.output.parent / "data_manifest.json"
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
