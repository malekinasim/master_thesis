#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


TASKS = ("mcq", "single")
PROBES = ("linsvm", "logreg")


def sanitize_model_name(model_dir_name: str) -> str:
    return model_dir_name.replace("__", "/")


def find_model_dirs(out_root: Path) -> list[Path]:
    reports_dir = out_root / "reports"
    if not reports_dir.exists():
        return []
    return sorted([p for p in reports_dir.iterdir() if p.is_dir()])


def read_probe_summary(
    out_root: Path,
    model_dir_name: str,
    task: str,
    probe: str,
) -> dict | None:
    tables_dir = (
        out_root
        / "reports"
        / model_dir_name
        / "robustness"
        / "local"
        / "probe_local"
        / task
        / probe
        / "tables"
    )

    auc_csv = tables_dir / "probe_robustness_per_layer.csv"
    acc_csv = tables_dir / "probe_acc_curves.csv"

    if not auc_csv.exists() or not acc_csv.exists():
        return None

    auc_df = pd.read_csv(auc_csv)
    acc_df = pd.read_csv(acc_csv)

    required_auc = {"layer", "robustness_auc"}
    required_acc = {"layer", "sigma", "acc"}

    if not required_auc.issubset(auc_df.columns):
        raise ValueError(f"{auc_csv} is missing columns: {required_auc - set(auc_df.columns)}")
    if not required_acc.issubset(acc_df.columns):
        raise ValueError(f"{acc_csv} is missing columns: {required_acc - set(acc_df.columns)}")

    auc_df = auc_df.sort_values("layer").reset_index(drop=True)
    acc_df = acc_df.sort_values(["layer", "sigma"]).reset_index(drop=True)

    # baseline at sigma = 0 on the last layer
    sigma0 = acc_df[acc_df["sigma"] == 0]
    if sigma0.empty:
        # fallback for float formatting issues
        sigma0 = acc_df[acc_df["sigma"].round(10) == 0]

    if sigma0.empty:
        acc0_last = None
    else:
        last_layer = int(sigma0["layer"].max())
        last_row = sigma0[sigma0["layer"] == last_layer].iloc[0]
        acc0_last = float(last_row["acc"])

    min_idx = auc_df["robustness_auc"].idxmin()
    max_idx = auc_df["robustness_auc"].idxmax()

    row = {
        "Model": sanitize_model_name(model_dir_name),
        "Task": task.upper(),
        "Probe": probe,
        "Acc@σ=0 (last)": acc0_last,
        "Mean AUC": float(auc_df["robustness_auc"].mean()),
        "Min AUC": float(auc_df.loc[min_idx, "robustness_auc"]),
        "Min layer": int(auc_df.loc[min_idx, "layer"]),
        "Max AUC": float(auc_df.loc[max_idx, "robustness_auc"]),
        "Max layer": int(auc_df.loc[max_idx, "layer"]),
    }
    return row


def build_summary_table(out_root: Path, models: list[str] | None = None) -> pd.DataFrame:
    if not models:
        model_dirs = [p.name for p in find_model_dirs(out_root)]
    else:
        model_dirs = models

    rows = []
    for model_dir_name in model_dirs:
        for task in TASKS:
            for probe in PROBES:
                row = read_probe_summary(out_root, model_dir_name, task, probe)
                if row is not None:
                    rows.append(row)

    if not rows:
        raise FileNotFoundError("No valid probe-space CSV pairs found under out/reports/...")

    df = pd.DataFrame(rows)

    # sort for thesis-friendly layout
    task_order = {"MCQ": 0, "SINGLE": 1}
    probe_order = {"linsvm": 0, "logreg": 1}

    df["_task_order"] = df["Task"].map(task_order)
    df["_probe_order"] = df["Probe"].map(probe_order)
    df = df.sort_values(["Task", "_task_order", "Model", "_probe_order"]).drop(
        columns=["_task_order", "_probe_order"]
    )

    # format numeric columns
    num_cols = ["Acc@σ=0 (last)", "Mean AUC", "Min AUC", "Max AUC"]
    for c in num_cols:
        df[c] = df[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Table 11 summary for probe-space robustness.")
    ap.add_argument("--out_root", required=True, help="Example: out")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model directory names under out/reports, e.g. Qwen__Qwen2.5-0.5B",
    )
    ap.add_argument(
        "--out_csv",
        default=None,
        help="Default: <out_root>/reports/tables/table11_probe_robustness_summary.csv",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else out_root / "reports" / "tables" / "table11_probe_robustness_summary.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = build_summary_table(out_root=out_root, models=args.models)
    df.to_csv(out_csv, index=False)

    print(f"[SAVE] {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()