#!/usr/bin/env python3
"""
Export + plot the relationship between "decision geometry" (RQ1 probes) and robustness (RQ3).

New in this version:
- Keeps the original single scatter plots
- Also creates paired two-panel scatter plots across models
  (e.g. DeepSeek on the left, Qwen on the right)
"""

from __future__ import annotations

import glob
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


LABELS = {
    "Qwen__Qwen2.5-0.5B": "Qwen2.5-0.5B",
    "deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-R1-Distill (1.5B)",
    "Ammar-alhaj-ali__DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-R1-Distill (1.5B)",
}


def pretty_model(model_path: str) -> str:
    return LABELS.get(model_path, model_path.replace("__", "/"))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def safe_filename(s: str) -> str:
    s = s.replace("/", "__").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-__" else "_" for ch in s)


def discover_models(out_root: str) -> List[str]:
    base = Path(out_root) / "reports"
    if not base.exists():
        return []
    return sorted([d.name for d in base.iterdir() if d.is_dir()])


def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })


# ----------------------------
# File finders
# ----------------------------
def find_rq1_metrics_csv(out_root: str, model_path: str, task: str, method: str) -> Optional[str]:
    candidates = [
        Path(out_root) / "reports" / model_path / "lin_probs" / "tables" / task / f"{task}_{method}_perlayer_metrics_by_split.csv",
        Path(out_root) / "reports" / model_path / "lin_probs" / "tables" / task / f"{task}_{method}_perlayer_metrics.csv",
        Path(out_root) / "reports" / model_path / "tables" / task / f"{task}_{method}_perlayer_metrics_by_split.csv",
        Path(out_root) / "reports" / model_path / "tables" / task / f"{task}_{method}_perlayer_metrics.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    patt = str(Path(out_root) / "reports" / model_path / "**" / f"{task}_{method}_perlayer_metrics*.csv")
    hits = glob.glob(patt, recursive=True)
    return hits[0] if hits else None


def find_probe_space_robustness_csv(out_root: str, model_path: str, task: str, method: str) -> Optional[str]:
    candidates = [
        Path(out_root) / "reports" / model_path / "robustness" / "local" / "probe_local" / task / method / "tables" / "probe_robustness_per_layer.csv",
        Path(out_root) / "reports" / model_path / "robustness" / "local" / "probe_local" / method / "tables" / "probe_robustness_per_layer.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    patt = str(Path(out_root) / "reports" / model_path / "**" / "probe_robustness_per_layer.csv")
    hits = glob.glob(patt, recursive=True)
    hits2 = [
        h for h in hits
        if (f"/{task}/" in h or f"\\{task}\\" in h) and (f"/{method}/" in h or f"\\{method}\\" in h)
    ]
    return hits2[0] if hits2 else (hits[0] if hits else None)


def find_robustness_csv(out_root: str, model_path: str, task: str, method: str, robustness_kind: str) -> Optional[str]:
    if robustness_kind == "probe_space":
        return find_probe_space_robustness_csv(out_root, model_path, task, method)

    patt = str(Path(out_root) / "reports" / model_path / "**" / f"*{robustness_kind}*{task}*per_layer*.csv")
    hits = glob.glob(patt, recursive=True)
    return hits[0] if hits else None


# ----------------------------
# Stats
# ----------------------------
def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    m = x.notna() & y.notna()
    if int(m.sum()) < 3:
        return float("nan")
    return float(x[m].corr(y[m], method="pearson"))


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    m = x.notna() & y.notna()
    if int(m.sum()) < 3:
        return float("nan")
    rx = x[m].rank(method="average")
    ry = y[m].rank(method="average")
    return float(rx.corr(ry, method="pearson"))


def residual_corr_vs_depth(df: pd.DataFrame, xcol: str, ycol: str, depth_col: str = "depth_norm") -> Tuple[float, float]:
    sub = df[[depth_col, xcol, ycol]].dropna()
    if len(sub) < 4:
        return float("nan"), float("nan")

    d = sub[depth_col].to_numpy()
    x = sub[xcol].to_numpy()
    y = sub[ycol].to_numpy()

    ax, bx = np.polyfit(d, x, deg=1)
    ay, by = np.polyfit(d, y, deg=1)
    x_res = x - (ax * d + bx)
    y_res = y - (ay * d + by)

    return spearman_corr(x_res, y_res), pearson_corr(x_res, y_res)


# ----------------------------
# Plotters
# ----------------------------
def plot_scatter(df: pd.DataFrame, xcol: str, title: str, out_png: str) -> None:
    apply_plot_style()

    rho = spearman_corr(df[xcol].to_numpy(), df["robustness_auc"].to_numpy())
    r = pearson_corr(df[xcol].to_numpy(), df["robustness_auc"].to_numpy())

    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    ax.scatter(df[xcol], df["robustness_auc"], s=22)

    ax.set_xlabel(xcol)
    ax.set_ylabel("Robustness AUC")
    ax.set_title(f"{title}\nSpearman ρ={rho:.3f} | Pearson r={r:.3f}", pad=10)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    ax.margins(x=0.06, y=0.08)

    plt.tight_layout()
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()


def plot_overlay_vs_layer(df: pd.DataFrame, xcol: str, title: str, out_png: str) -> None:
    apply_plot_style()

    df = df.sort_values("layer").copy()
    x = df["layer"].to_numpy()

    a = df[xcol].to_numpy(dtype=float)
    b = df["robustness_auc"].to_numpy(dtype=float)

    def minmax(v):
        v = pd.Series(v)
        if v.dropna().empty:
            return v.to_numpy()
        mn = float(v.min())
        mx = float(v.max())
        if mx - mn < 1e-12:
            return (v * 0.0 + 0.5).to_numpy()
        return ((v - mn) / (mx - mn)).to_numpy()

    aN = minmax(a)
    bN = minmax(b)

    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    ax.plot(x, aN, linewidth=2.2, label=f"{xcol} (min-max)")
    ax.plot(x, bN, linewidth=2.2, label="robustness_auc (min-max)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized value")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)

    plt.tight_layout()
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()


def plot_scatter_pair_across_models(
    pair_data: list[tuple[str, pd.DataFrame]],
    task: str,
    method: str,
    xcol: str,
    out_png: str,
) -> None:
    """
    Create a two-panel scatter plot across models for one task+method+x_metric.
    Example:
      left  = DeepSeek
      right = Qwen
    """
    apply_plot_style()

    if len(pair_data) < 2:
        return

    # keep only first two models for the paired figure
    pair_data = pair_data[:2]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharey=True)

    global_xmin = min(float(df[xcol].min()) for _, df in pair_data if not df[xcol].dropna().empty)
    global_xmax = max(float(df[xcol].max()) for _, df in pair_data if not df[xcol].dropna().empty)
    global_ymin = min(float(df["robustness_auc"].min()) for _, df in pair_data if not df["robustness_auc"].dropna().empty)
    global_ymax = max(float(df["robustness_auc"].max()) for _, df in pair_data if not df["robustness_auc"].dropna().empty)

    xpad = max(0.02 * (global_xmax - global_xmin), 1e-6)
    ypad = max(0.04 * (global_ymax - global_ymin), 1e-6)

    for ax, (model_label, df) in zip(axes, pair_data):
        rho = spearman_corr(df[xcol].to_numpy(), df["robustness_auc"].to_numpy())
        r = pearson_corr(df[xcol].to_numpy(), df["robustness_auc"].to_numpy())

        ax.scatter(df[xcol], df["robustness_auc"], s=22)

        ax.set_title(
            f"{model_label} | {task.upper()} | {method.upper()}\n"
            f"Spearman ρ={rho:.3f} | Pearson r={r:.3f}",
            fontsize=11,
            pad=8
        )
        ax.set_xlabel(xcol)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", labelrotation=0)
        ax.tick_params(axis="y", labelrotation=0)
        ax.set_xlim(global_xmin - xpad, global_xmax + xpad)
        ax.set_ylim(global_ymin - ypad, global_ymax + ypad)

    axes[0].set_ylabel("Robustness AUC")

    fig.suptitle(
        f"Relationship between {xcol} and probe-space robustness across layers",
        y=1.02
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()


# ----------------------------
# Loaders / merge
# ----------------------------
def load_rq1_metrics(rq1_csv: str, split: str = "test") -> pd.DataFrame:
    df = pd.read_csv(rq1_csv)
    df = normalize_columns(df)

    if "layer" not in df.columns:
        raise ValueError(f"RQ1 metrics missing 'layer': {rq1_csv}")
    df["layer"] = df["layer"].astype(int)

    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.lower()
        df = df[df["split"] == split.lower()].copy()

    for c in ["acc", "auroc", "mean_margin"]:
        if c not in df.columns:
            df[c] = np.nan

    return df.sort_values("layer")


def load_robustness_per_layer(rob_csv: str) -> pd.DataFrame:
    df = pd.read_csv(rob_csv)
    df = normalize_columns(df)

    if "layer" not in df.columns:
        raise ValueError(f"Robustness file missing 'layer': {rob_csv}")
    df["layer"] = df["layer"].astype(int)

    rcol = None
    for c in ["robustness_auc", "robustness", "auc", "auc_norm", "robust_auc"]:
        if c in df.columns:
            rcol = c
            break
    if rcol is None:
        raise ValueError(f"Robustness file missing robustness column: {rob_csv} (cols={df.columns.tolist()})")

    df = df.rename(columns={rcol: "robustness_auc"})
    return df[["layer", "robustness_auc"]].sort_values("layer")


def merge_geometry_and_robustness(rq1_df: pd.DataFrame, rob_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(rq1_df, rob_df, on="layer", how="inner").sort_values("layer")
    last_layer = int(merged["layer"].max()) if len(merged) else 0
    merged["depth_norm"] = merged["layer"] / last_layer if last_layer > 0 else 0.0
    return merged


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser("Export + plot geometry↔robustness relationships")
    ap.add_argument("--out_root", default="out", help="Root output folder (contains reports/)")
    ap.add_argument("--models", default="auto",
                    help="Comma list of HF model names OR model_path (with __). Use 'auto' to discover from out/reports/*")
    ap.add_argument("--tasks", default="mcq,single", help="Comma list: mcq,single")
    ap.add_argument("--methods", default="massmean,lda,logreg,linsvm", help="Comma list: massmean,lda,logreg,linsvm")
    ap.add_argument("--robustness_kind", default="probe_space", help="Recommended: probe_space")
    ap.add_argument("--split", default="test", help="If RQ1 file has split column: which split to use.")
    ap.add_argument("--x_metrics", default="mean_margin,auroc", help="Comma list: mean_margin,auroc,acc")
    ap.add_argument("--make_plots", action="store_true", help="Generate single scatter plots per combo")
    ap.add_argument("--make_overlay", action="store_true", help="Generate overlay-vs-layer plots per combo")
    ap.add_argument("--make_paired_plots", action="store_true",
                    help="Generate paired two-panel scatter plots across models")
    ap.add_argument("--min_points", type=int, default=6, help="Min merged layers to keep a combo")
    ap.add_argument("--out_dir", default=None,
                    help="Override analysis output dir (default: out_root/analysis/geom_vs_robustness)")
    args = ap.parse_args()

    out_root = args.out_root
    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    x_metrics = [m.strip().lower() for m in args.x_metrics.split(",") if m.strip()]
    robustness_kind = args.robustness_kind

    if args.models.strip().lower() == "auto":
        model_paths = discover_models(out_root)
        if not model_paths:
            raise FileNotFoundError(f"No models discovered under: {Path(out_root) / 'reports'}")
    else:
        raw = [m.strip() for m in args.models.split(",") if m.strip()]
        model_paths = [m.replace("/", "__") for m in raw]

    analysis_dir = args.out_dir or str(Path(out_root) / "analysis" / "geom_vs_robustness")
    ensure_dir(analysis_dir)

    long_rows = []
    summary_rows = []

    # collect merged data for paired figures
    paired_store: dict[tuple[str, str, str], list[tuple[str, pd.DataFrame]]] = {}

    for model_path in model_paths:
        for task in tasks:
            for method in methods:
                rq1_csv = find_rq1_metrics_csv(out_root, model_path, task, method)
                rob_csv = find_robustness_csv(out_root, model_path, task, method, robustness_kind)

                if not rq1_csv or not rob_csv:
                    continue

                try:
                    rq1_df = load_rq1_metrics(rq1_csv, split=args.split)
                    rob_df = load_robustness_per_layer(rob_csv)
                    merged = merge_geometry_and_robustness(rq1_df, rob_df)
                except Exception as e:
                    print(f"[WARN] Skip {model_path} {task} {method}: {e}")
                    continue

                if len(merged) < args.min_points:
                    continue

                combo_id = f"{model_path}__{task}__{method}"
                combo_dir = Path(analysis_dir) / "per_combo" / task / method
                ensure_dir(combo_dir)

                merged_out_csv = combo_dir / f"{safe_filename(combo_id)}__merged.csv"
                merged.to_csv(merged_out_csv, index=False)

                for xcol in x_metrics:
                    if xcol not in merged.columns or merged[xcol].dropna().empty:
                        continue

                    rho = spearman_corr(merged[xcol].to_numpy(), merged["robustness_auc"].to_numpy())
                    r = pearson_corr(merged[xcol].to_numpy(), merged["robustness_auc"].to_numpy())
                    rho_res, r_res = residual_corr_vs_depth(merged, xcol=xcol, ycol="robustness_auc", depth_col="depth_norm")

                    peak_x_layer = int(merged.loc[merged[xcol].idxmax(), "layer"]) if merged[xcol].notna().any() else None
                    peak_r_layer = int(merged.loc[merged["robustness_auc"].idxmax(), "layer"]) if merged["robustness_auc"].notna().any() else None

                    summary_rows.append({
                        "model_path": model_path,
                        "model_label": pretty_model(model_path),
                        "task": task,
                        "method": method,
                        "robustness_kind": robustness_kind,
                        "x_metric": xcol,
                        "n_layers_merged": int(len(merged)),
                        "spearman_rho": rho,
                        "pearson_r": r,
                        "spearman_rho_resid_depth": rho_res,
                        "pearson_r_resid_depth": r_res,
                        "peak_x_layer": peak_x_layer,
                        "peak_robust_layer": peak_r_layer,
                        "peak_layer_diff(robust-x)": (peak_r_layer - peak_x_layer) if (peak_r_layer is not None and peak_x_layer is not None) else None,
                        "rq1_csv": rq1_csv,
                        "rob_csv": rob_csv,
                        "merged_csv": str(merged_out_csv),
                    })

                    if args.make_plots:
                        title = f"{pretty_model(model_path)} | {task.upper()} | {method.upper()} | {xcol} vs robustness"
                        out_png = combo_dir / f"{safe_filename(combo_id)}__scatter__{xcol}.png"
                        plot_scatter(merged, xcol=xcol, title=title, out_png=str(out_png))

                    if args.make_overlay:
                        title = f"{pretty_model(model_path)} | {task.upper()} | {method.upper()} | layer-wise alignment"
                        out_png = combo_dir / f"{safe_filename(combo_id)}__overlay__{xcol}.png"
                        plot_overlay_vs_layer(merged, xcol=xcol, title=title, out_png=str(out_png))

                    key = (task, method, xcol)
                    paired_store.setdefault(key, []).append((pretty_model(model_path), merged.copy()))

                merged2 = merged.copy()
                merged2.insert(0, "model_path", model_path)
                merged2.insert(1, "model_label", pretty_model(model_path))
                merged2.insert(2, "task", task)
                merged2.insert(3, "method", method)
                merged2.insert(4, "robustness_kind", robustness_kind)
                long_rows.append(merged2)

    if not long_rows:
        raise RuntimeError(
            "No combinations found.\n"
            "Check:\n"
            "  1) out_root points to your out folder,\n"
            "  2) RQ1 metrics exist (perlayer_metrics*.csv),\n"
            "  3) probe-space robustness per layer exists (probe_robustness_per_layer.csv).\n"
        )

    long_df = pd.concat(long_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["task", "method", "x_metric", "spearman_rho"], ascending=[True, True, True, False]
    )

    out_long_csv = Path(analysis_dir) / "geometry_vs_robustness_long.csv"
    out_sum_csv = Path(analysis_dir) / "geometry_vs_robustness_summary.csv"
    long_df.to_csv(out_long_csv, index=False)
    summary_df.to_csv(out_sum_csv, index=False)
    print(f"[SAVE] {out_long_csv}")
    print(f"[SAVE] {out_sum_csv}")

    out_xlsx = Path(analysis_dir) / "geometry_vs_robustness.xlsx"
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            long_df.to_excel(w, sheet_name="long", index=False)
            summary_df.to_excel(w, sheet_name="summary", index=False)
        print(f"[SAVE] {out_xlsx}")
    except Exception as e:
        print(f"[WARN] Could not write Excel ({out_xlsx}): {e}")
        print("       CSV outputs are still available.")

    # paired plots
    if args.make_paired_plots:
        paired_dir = Path(analysis_dir) / "paired_plots"
        ensure_dir(paired_dir)

        for (task, method, xcol), items in paired_store.items():
            if len(items) < 2:
                continue

            out_png = paired_dir / f"{safe_filename(task)}__{safe_filename(method)}__{safe_filename(xcol)}__paired_scatter.png"
            plot_scatter_pair_across_models(
                pair_data=items,
                task=task,
                method=method,
                xcol=xcol,
                out_png=str(out_png),
            )
            print(f"[SAVE] {out_png}")


if __name__ == "__main__":
    main()