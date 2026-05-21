#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_thr(path: str) -> float:
    # expects ...thr0p9... or thr0p85... or thr0p8...
    m = re.search(r"thr0p(\d+)", Path(path).name)
    if not m:
        raise ValueError(f"Cannot parse threshold from filename: {path}")
    digits = m.group(1)  # "9", "85", "8"
    return float("0." + digits)


def find_col(df: pd.DataFrame, key: str) -> str:
    """Find a column by fuzzy matching (case-insensitive)."""
    key_l = key.lower()
    cols = [c for c in df.columns if key_l in c.lower()]
    if not cols:
        raise ValueError(f"Could not find a column containing '{key}'. Columns={df.columns.tolist()}")
    return cols[0]


def normalise_task(x: str) -> str:
    x = str(x).strip().lower()
    if x in {"mcq", "multiple-choice", "multiple choice"}:
        return "mcq"
    if x in {"single", "single-token", "single token"}:
        return "single"
    return x


def make_window(start, end) -> str:
    s = str(start).strip()
    e = str(end).strip()
    if s.upper() == "N/A" or s == "" or s.lower() == "nan":
        return "N/A"
    if e.upper() == "N/A" or e == "" or e.lower() == "nan":
        return str(s)
    return f"{s}–{e}"  # en-dash

def main(): 
    ap = argparse.ArgumentParser("Create EDL threshold-sensitivity tables and plots (all probe families)")
    ap.add_argument("--csvs", required=True,
                    help="Comma-separated CSVs for thr0p9/thr0p85/thr0p8 (k=2), e.g. "
                         "table_edl_all_tasks_all_methods_thr0p9_k2.csv,...")
    ap.add_argument("--out_root", default="out", help="Project out/ directory")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_tables = out_root / "reports" / "tables"
    out_figs = out_root / "reports" / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    paths = [p.strip() for p in args.csvs.split(",") if p.strip()]
    if len(paths) < 2:
        raise SystemExit("Provide at least two threshold CSVs.")

    frames = []
    for p in paths:
        thr = parse_thr(p)
        path=out_tables / p
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        task_col = find_col(df, "Task")
        method_col = find_col(df, "Method")
        model_col = find_col(df, "Model")
        start_col = find_col(df, "EDL start")
        end_col = find_col(df, "EDL end")

        tmp = df.copy()
        tmp["threshold"] = thr
        tmp["Task"] = tmp[task_col].map(normalise_task)
        tmp["Method"] = tmp[method_col].astype(str).str.upper().str.strip()
        tmp["Model"] = tmp[model_col].astype(str).str.strip()

        tmp["edl_window"] = [
            make_window(s, e) for s, e in zip(tmp[start_col], tmp[end_col])
        ]
        tmp["defined"] = tmp["edl_window"].astype(str).str.upper().ne("N/A")

        # Keep optional descriptive columns if present (not per-threshold)
        keep = ["Task", "Method", "Model", "threshold", "edl_window", "defined"]
        # If your file has these, keep them:
        for opt in ["Layers", "Peak AUROC", "Peak layer", "ACC at peak"]:
            if any(opt.lower() in c.lower() for c in df.columns):
                keep.append(find_col(tmp, opt))

        tmp = tmp[keep]
        frames.append(tmp)

    big = pd.concat(frames, ignore_index=True)

    # ---------- Coverage table (how many models defined per task/method/threshold)
    cov = (
        big.groupby(["Task", "Method", "threshold"], as_index=False)
           .agg(n_defined=("defined", "sum"), n_total=("defined", "count"))
    )
    cov["n_na"] = cov["n_total"] - cov["n_defined"]
    cov = cov.sort_values(["Task", "Method", "threshold"])
    cov.to_csv(out_tables / "table_edl_threshold_coverage_all_methods.csv", index=False)

    # ---------- Wide tables: one per task, columns are thresholds (window strings)
    for task in ["mcq", "single"]:
        sub = big[big["Task"] == task].copy()
        if sub.empty:
            continue

        wide = (
            sub.pivot_table(index=["Model", "Method"], columns="threshold", values="edl_window",
                            aggfunc="first")
            .reset_index()
        )

        # rename threshold columns nicely
        new_cols = {}
        for c in wide.columns:
            if isinstance(c, float) or isinstance(c, int):
                new_cols[c] = f"EDL window (τ={c:.2f})"
        wide = wide.rename(columns=new_cols)

        # optional: attach peak metrics once (take from highest threshold file if present)
        # (Peak AUROC etc are independent of threshold, but they are repeated in each input file.)
        metrics_cols = [c for c in sub.columns if c in ["Layers", "Peak AUROC", "Peak layer", "ACC at peak"]]
        if metrics_cols:
            base_metrics = (
                sub.sort_values("threshold", ascending=False)
                   .drop_duplicates(subset=["Model", "Method"])[["Model", "Method"] + metrics_cols]
            )
            wide = wide.merge(base_metrics, on=["Model", "Method"], how="left")

        out_path = out_tables / f"table_edl_windows_by_threshold_{task}_k2.csv"
        wide.to_csv(out_path, index=False)

    # ---------- Plots: coverage vs threshold, one plot per task
    for task in ["mcq", "single"]:
        sub = cov[cov["Task"] == task].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(8.5, 5))
        for method in sorted(sub["Method"].unique()):
            ss = sub[sub["Method"] == method].sort_values("threshold")
            plt.plot(ss["threshold"].values, ss["n_defined"].values, marker="o", label=method)

        plt.xlabel("AUROC threshold (τ)")
        plt.ylabel("Number of defined EDL windows (out of 6 models)")
        plt.title(f"{task.upper()} – EDL coverage vs threshold (K = 2)")
        plt.grid(alpha=0.25)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
        plt.tight_layout()
        plt.savefig(out_figs / f"fig_edl_coverage_vs_threshold_{task}_k2.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("[OK] Saved tables and figures under:")
    print(f"  Tables:  {out_tables}")
    print(f"  Figures: {out_figs}")


if __name__ == "__main__":
    main()