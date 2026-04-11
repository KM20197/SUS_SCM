#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute within-run equivalence tests (TOST) and minimum detectable effects (MDE)
from Code4 raw_replications.csv files.

Design constraints:
- Compute tests *within* a single run, aligned by replication index.
- Do NOT attempt "paired across runs" comparisons.

Input schemas supported:
- horizon_days or horizon
- replication or replication_id

Outputs:
- tost_mde_withinrun.csv (one row per run_id × scenario × horizon_days × metric × comparison)
- run_manifest.json (minimal provenance)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_METRICS = [
    "total_cost",
    "ordering_cost",
    "stockout_cost",
    "routing_cost",
    "fill_rate",
]

DEFAULT_BASELINES = ["BASESTOCK", "DYN_FC"]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _canon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize raw_replications schemas across runs.

    Canonical columns used downstream:
      - horizon_days
      - replication
    """
    df = df.copy()
    if "horizon" in df.columns and "horizon_days" not in df.columns:
        df = df.rename(columns={"horizon": "horizon_days"})
    if "replication_id" in df.columns and "replication" not in df.columns:
        df = df.rename(columns={"replication_id": "replication"})
    # keep numeric types stable
    if "horizon_days" in df.columns:
        df["horizon_days"] = pd.to_numeric(df["horizon_days"], errors="coerce")
    if "replication" in df.columns:
        df["replication"] = pd.to_numeric(df["replication"], errors="coerce")
    # normalize optional naming variants
    if "reward_modes" in df.columns and "reward_mode" not in df.columns:
        df = df.rename(columns={"reward_modes": "reward_mode"})
    if "ga_modes" in df.columns and "ga_mode" not in df.columns:
        df = df.rename(columns={"ga_modes": "ga_mode"})
    return df


def _tost_pvalues(d: np.ndarray, margin: float) -> Tuple[float, float, float, bool]:
    """
    Two one-sided t-tests for equivalence of mean(d) to 0 within [-margin, +margin].
    Returns (p1, p2, p_equiv, equiv_at_alpha) for alpha handled by caller.
    """
    n = int(len(d))
    if n < 3:
        return (float("nan"), float("nan"), float("nan"), False)
    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        p1 = 0.0 if md > -margin else 1.0
        p2 = 0.0 if md < margin else 1.0
        return (p1, p2, max(p1, p2), False)
    se = sd / math.sqrt(n)
    dfree = n - 1
    # H0: md <= -margin; Ha: md > -margin
    t1 = (md + margin) / se
    p1 = 1.0 - stats.t.cdf(t1, dfree)
    # H0: md >= +margin; Ha: md < +margin
    t2 = (md - margin) / se
    p2 = stats.t.cdf(t2, dfree)
    return (float(p1), float(p2), float(max(p1, p2)), False)


def _mde(sd: float, n: int, alpha: float, power: float) -> Tuple[float, float]:
    """
    Paired t-test MDE (two-sided) under normal approximation for power:
      MDE_abs = (t_{1-alpha/2, df} + z_{power}) * sd / sqrt(n)
      MDE_dz  = (t_{1-alpha/2, df} + z_{power}) / sqrt(n)
    """
    dfree = n - 1
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, dfree))
    zpow = float(stats.norm.ppf(power))
    if not np.isfinite(sd) or sd <= 0.0:
        return (float("nan"), float("nan"))
    mde_abs = (tcrit + zpow) * sd / math.sqrt(n)
    mde_dz = (tcrit + zpow) / math.sqrt(n)
    return (float(mde_abs), float(mde_dz))


def compute_withinrun(df: pd.DataFrame,
                      run_id: str,
                      metrics: List[str],
                      baselines: List[str],
                      alpha: float,
                      power: float,
                      cost_rel_margin: float,
                      fill_abs_margin: float) -> pd.DataFrame:
    df = _canon(df)
    if "run_id" not in df.columns:
        df["run_id"] = run_id

    required = ["scenario", "horizon_days", "forecaster", "reward_mode", "ga_mode", "replication", "policy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}; available={list(df.columns)}")

    keys = ["run_id", "scenario", "horizon_days", "forecaster", "reward_mode", "ga_mode"]

    rows: List[Dict] = []
    for gk, gdf in df.groupby(keys, dropna=False):
        for base in baselines:
            sub = gdf[gdf["policy"].isin(["AI", base])]
            if sub.empty:
                continue
            piv = sub.pivot_table(index="replication", columns="policy", values=metrics, aggfunc="mean")
            for metric in metrics:
                if (metric, "AI") not in piv.columns or (metric, base) not in piv.columns:
                    continue
                x = piv[(metric, "AI")].dropna()
                y = piv[(metric, base)].dropna()
                idx = x.index.intersection(y.index)
                if len(idx) < 3:
                    continue
                d = (x.loc[idx] - y.loc[idx]).astype(float).to_numpy()
                n = int(len(d))
                md = float(np.mean(d))
                sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
                se = sd / math.sqrt(n) if np.isfinite(sd) and sd > 0 else float("inf")
                dfree = n - 1

                # equivalence margin
                if metric == "fill_rate":
                    margin = float(fill_abs_margin)
                else:
                    baseline_mean = float(np.mean(y.loc[idx].to_numpy()))
                    margin = float(cost_rel_margin * abs(baseline_mean))

                p1, p2, p_equiv, _ = _tost_pvalues(d, margin)
                equiv = bool(np.isfinite(p_equiv) and (p_equiv < alpha))

                # CI for mean difference (t)
                if np.isfinite(se) and se != float("inf"):
                    tci = float(stats.t.ppf(1.0 - alpha / 2.0, dfree))
                    ci_lo = md - tci * se
                    ci_hi = md + tci * se
                else:
                    ci_lo, ci_hi = (float("nan"), float("nan"))

                # MDE
                mde_abs, mde_dz = _mde(sd, n, alpha, power)

                dz = md / sd if np.isfinite(sd) and sd > 0 else float("nan")

                rows.append({
                    "run_id": gk[0],
                    "scenario": gk[1],
                    "horizon_days": int(gk[2]) if pd.notna(gk[2]) else None,
                    "forecaster": gk[3],
                    "reward_mode": gk[4],
                    "ga_mode": gk[5],
                    "metric": metric,
                    "comparison": f"AI_vs_{base}",
                    "n": n,
                    "mean_diff": md,
                    "sd_diff": sd,
                    "dz": dz,
                    "ci95_lo": ci_lo,
                    "ci95_hi": ci_hi,
                    "equiv_margin": margin,
                    "tost_p1": p1,
                    "tost_p2": p2,
                    "tost_p_equiv": p_equiv,
                    "equiv_at_alpha": equiv,
                    "mde_abs": mde_abs,
                    "mde_dz": mde_dz,
                })

    return pd.DataFrame(rows)


def _parse_inputs(inputs: List[str]) -> List[Tuple[str, Path]]:
    parsed = []
    for s in inputs:
        if "::" in s:
            rid, p = s.split("::", 1)
            parsed.append((rid.strip(), Path(p.strip())))
        else:
            p = Path(s.strip())
            rid = p.parent.parent.name if p.name.startswith("raw_replications") else p.stem
            parsed.append((rid, p))
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more items: RUN_ID::/path/to/raw_replications.csv (or just a path).")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--cost_rel_margin", type=float, default=0.05,
                    help="Equivalence margin for cost metrics as a fraction of baseline mean (default 0.05).")
    ap.add_argument("--fill_abs_margin", type=float, default=0.01,
                    help="Equivalence margin for fill_rate in absolute units (default 0.01 = 1pp).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = _parse_inputs(args.inputs)

    all_rows = []
    for rid, path in inputs:
        df = _read_csv(path)
        res = compute_withinrun(
            df=df,
            run_id=rid,
            metrics=args.metrics,
            baselines=args.baselines,
            alpha=args.alpha,
            power=args.power,
            cost_rel_margin=args.cost_rel_margin,
            fill_abs_margin=args.fill_abs_margin,
        )
        all_rows.append(res)

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    out_csv = out_dir / "tost_mde_withinrun.csv"
    out.to_csv(out_csv, index=False)

    manifest = {
        "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "inputs": [{"run_id": rid, "path": str(path)} for rid, path in inputs],
        "alpha": args.alpha,
        "power": args.power,
        "metrics": args.metrics,
        "baselines": args.baselines,
        "cost_rel_margin": args.cost_rel_margin,
        "fill_abs_margin": args.fill_abs_margin,
        "output": str(out_csv),
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"WROTE {out_csv} rows={len(out)}")
    print(f"WROTE {out_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    import datetime  # keep local to avoid linter complaints in minimal environments
    main()
