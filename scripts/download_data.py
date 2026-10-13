"""Download the UCI Chronic Kidney Disease dataset (id 336) to data/kidney_disease.csv.

Source: Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). Chronic Kidney Disease
[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020
License: CC BY 4.0

The repo already ships the converted CSV (400 rows, CC BY 4.0), so you only
need this script to rebuild it from the original source.

The official download is a zip containing a RAR archive, and the ARFF inside has
formatting errors (extra commas, tabs), which is why generic loaders such as
``ucimlrepo`` fail on it. This script extracts the RAR with ``bsdtar`` (built in
on macOS as ``tar``; ``apt install libarchive-tools`` on Linux) and parses the
ARFF itself.
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/336/chronic+kidney+disease.zip"
OUT = Path(__file__).resolve().parents[1] / "data" / "kidney_disease.csv"


def _ssl_context() -> ssl.SSLContext:
    # python.org builds on macOS ship without system certificates; certifi fixes that
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _extract_rar(data: bytes) -> str:
    tool = shutil.which("bsdtar") or (shutil.which("tar") if sys.platform == "darwin" else None)
    if tool is None:
        sys.exit("Need bsdtar to open the RAR archive: on Linux run `apt install libarchive-tools`.")
    with tempfile.TemporaryDirectory() as tmp:
        rar = Path(tmp) / "ckd.rar"
        rar.write_bytes(data)
        subprocess.run([tool, "-xf", str(rar), "-C", tmp], check=True)
        arffs = sorted(Path(tmp).rglob("*.arff"))
        path = next((p for p in arffs if "full" in p.name.lower()), arffs[0])
        return path.read_text(encoding="utf-8", errors="replace")


def read_arff_from_zip(payload: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
        arffs = [n for n in names if n.lower().endswith(".arff")]
        if arffs:
            name = next((n for n in arffs if "full" in n.lower()), arffs[0])
            return zf.read(name).decode("utf-8", errors="replace")
        rars = [n for n in names if n.lower().endswith(".rar")]
        if rars:
            return _extract_rar(zf.read(rars[0]))
    sys.exit(f"No ARFF or RAR file found in the download: {names}")


def parse_arff(text: str) -> tuple[list[str], list[list[str]]]:
    names: list[str] = []
    rows: list[list[str]] = []
    skipped = 0
    in_data = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        low = line.lower()
        if not in_data:
            if low.startswith("@attribute"):
                m = re.match(r"@attribute\s+'?([^'\s]+)'?", line, flags=re.IGNORECASE)
                names.append(m.group(1))
            elif low.startswith("@data"):
                in_data = True
            continue
        values = [v.strip() for v in line.split(",")]
        if len(values) > len(names):
            # stray commas: drop empty tokens until the row fits
            values = [v for v in values if v != ""]
        if len(values) != len(names):
            skipped += 1
            continue
        rows.append(values)
    if skipped:
        print(f"Skipped {skipped} malformed rows")
    return names, rows


def main() -> None:
    if OUT.exists():
        print(f"{OUT} already exists, skipping download")
        return
    print(f"Downloading {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=60, context=_ssl_context()) as resp:
            payload = resp.read()
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"Download failed: {exc}\n"
            "Download the dataset manually from https://archive.ics.uci.edu/dataset/336 "
            "or Kaggle, and save it as data/kidney_disease.csv"
        )

    text = read_arff_from_zip(payload)

    names, rows = parse_arff(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows x {len(names)} columns to {OUT}")


if __name__ == "__main__":
    main()
