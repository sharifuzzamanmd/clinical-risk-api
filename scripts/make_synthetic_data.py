"""Generate a synthetic CKD-like dataset with the raw (UCI/Kaggle) column names.

Used by the test suite and CI so they never depend on downloading real data.
The numbers are invented; never report results trained on this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int = 400, seed: int = 0, positive_rate: float = 0.62) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ckd = rng.random(n) < positive_rate

    def pick(p_yes_ckd: float, p_yes_healthy: float, yes: str, no: str) -> np.ndarray:
        p = np.where(ckd, p_yes_ckd, p_yes_healthy)
        return np.where(rng.random(n) < p, yes, no)

    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "age": rng.normal(np.where(ckd, 55, 46), 15).clip(2, 90).round(),
            "bp": rng.choice([60, 70, 80, 90, 100], n),
            "sg": np.where(ckd, rng.choice([1.005, 1.010, 1.015], n), rng.choice([1.020, 1.025], n)),
            "al": np.where(ckd, rng.integers(0, 5, n), 0),
            "su": np.where(ckd, rng.integers(0, 4, n), 0),
            "rbc": pick(0.3, 0.02, "abnormal", "normal"),
            "pc": pick(0.35, 0.02, "abnormal", "normal"),
            "pcc": pick(0.15, 0.01, "present", "notpresent"),
            "ba": pick(0.1, 0.01, "present", "notpresent"),
            "bgr": rng.normal(np.where(ckd, 170, 110), 40).clip(50, 490).round(),
            "bu": rng.normal(np.where(ckd, 70, 33), 20).clip(5, 390).round(),
            "sc": rng.normal(np.where(ckd, 4.0, 0.9), np.where(ckd, 2.5, 0.3)).clip(0.4, 76).round(1),
            "sod": rng.normal(np.where(ckd, 133, 141), 5).round(),
            "pot": rng.normal(np.where(ckd, 4.9, 4.3), 0.6).round(1),
            "hemo": rng.normal(np.where(ckd, 10.6, 15.2), 1.5).round(1),
            "pcv": rng.normal(np.where(ckd, 33, 46), 5).round().astype(int).astype(str),
            "wc": rng.normal(np.where(ckd, 9000, 7700), 2000).round(-2).astype(int).astype(str),
            "rc": rng.normal(np.where(ckd, 3.9, 5.4), 0.6).round(1).astype(str),
            "htn": pick(0.6, 0.02, "yes", "no"),
            "dm": pick(0.55, 0.02, "yes", "no"),
            "cad": pick(0.12, 0.01, "yes", "no"),
            "appet": pick(0.3, 0.02, "poor", "good"),
            "pe": pick(0.3, 0.02, "yes", "no"),
            "ane": pick(0.25, 0.02, "yes", "no"),
            "classification": np.where(ckd, "ckd", "notckd"),
        }
    )
    # mimic the real file's quirks: missing values, stray tabs, '?' entries
    for col in ["rbc", "pc", "sod", "pot", "pcv", "wc", "rc", "bgr", "su"]:
        df.loc[rng.random(n) < 0.15, col] = np.nan
    df.loc[rng.random(n) < 0.03, "dm"] = "\tyes"
    df.loc[rng.random(n) < 0.03, "pcv"] = "\t?"
    df.loc[rng.random(n) < 0.01, "classification"] = "ckd\t"
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_kidney_disease.csv")
    parser.add_argument("-n", type=int, default=400)
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    make(args.n).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
