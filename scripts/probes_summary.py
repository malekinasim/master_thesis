import os
import glob
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# ensure_dir fallback
try:
    from src.io import ensure_dir
except Exception:
    def ensure_dir(p: str):
        os.makedirs(p, exist_ok=True)

CHANCE = {"mcq": 0.25, "single": 0.5}
DELTA = 0.02

# Legend labels for models
LABELS = {
    "EleutherAI__gpt-neo-125M": "GPT-Neo-125M",
    "Qwen__Qwen2.5-0.5B": "Qwen2.5-0.5B",
    "TinyLlama__TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
    "deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-R1-Distill (1.5B)",
    "facebook__opt-125m": "OPT-125M",
    "meta-llama__Llama-3.2-1B": "Llama-3.2-1B",
}

def pretty_model(model_path: str) -> str:
    return LABELS.get(model_path, model_path.replace("__", "/"))

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column names + ensures required metrics exist.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "layer" not in df.columns:
        raise ValueError(f"Missing 'layer' column. Columns={df.columns.tolist()}")
    df["layer"] = df["layer"].astype(int)
    for c in ["acc", "auroc", "mean_margin"]:
        if c not in df.columns:
            df[c] = np.nan
    return df.sort_values("layer")

def edl_probe_margin_acc(df: pd.DataFrame, task: str, k: int = 2):
    """
    Your EDL definition:
      mean_margin > 0 AND acc >= chance + DELTA
    for k consecutive layers.
    """
    df = df.sort_values("layer")
    ok = (df["mean_margin"].fillna(0) > 0) & (df["acc"].fillna(0) >= CHANCE[task] + DELTA)
    idx = ok.rolling(k).apply(lambda s: s.all(), raw=True).to_numpy()
    pos = np.where(idx == 1)[0]
    return int(df.iloc[pos[0]]["layer"]) if len(pos) else None

def read_metrics_csv(out_root: str, model_path: str, task: str, method: str) -> str | None:
    """
    Supports both:
      out/reports/<model>/tables/<task>/<task>_<method>_perlayer_metrics.csv
      out/reports/<model>/lin_probs/tables/<task>/<task>_<method>_perlayer_metrics.csv
    + recursive fallback.
    """
    p1 = os.path.join(out_root, "reports", model_path, "tables", task,
                      f"{task}_{method}_perlayer_metrics.csv")
    p2 = os.path.join(out_root, "reports", model_path, "lin_probs", "tables", task,
                      f"{task}_{method}_perlayer_metrics.csv")

    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2

    patt = os.path.join(out_root, "reports", model_path, "**", f"{task}_{method}_perlayer_metrics.csv")
    hits = glob.glob(patt, recursive=True)
    return hits[0] if hits else None

def plot_across_depth_all_models_v1(curves, metric: str, title: str, out_png: str, normalize_depth=True):
    """
    One curve per model, fixed (task, method).
    """
    plt.figure(figsize=(8.2, 5.))
    for model_path, df in curves:
        df = normalize_df(df)
        last_layer = int(df["layer"].max())
        x = (df["layer"] / last_layer) if (normalize_depth and last_layer > 0) else df["layer"]
        plt.plot(x, df[metric], label=pretty_model(model_path))

    plt.title(title)
    plt.xlabel("Normalized depth (layer / last_layer)" if normalize_depth else "Layer")
    plt.ylabel(metric.upper() if metric != "mean_margin" else "Mean margin")
    if metric in ("auroc", "acc"):
        plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.25)
    #plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_png}")

def plot_across_depth_all_models(curves, metric: str, title: str, out_png: str, normalize_depth=True):
    """
    One curve per model, fixed (task, method).
    """
    plt.figure(figsize=(8.2, 5.8))
    for model_path, df in curves:
        df = normalize_df(df)
        last_layer = int(df["layer"].max())
        x = (df["layer"] / last_layer) if (normalize_depth and last_layer > 0) else df["layer"]
        plt.plot(x, df[metric], label=pretty_model(model_path), linewidth=2.2)

    # Keep title minimal for thesis figures; captions already explain the figure
    if title:
        plt.title(title, fontsize=12)

    plt.xlabel("Normalized depth (layer / last_layer)" if normalize_depth else "Layer", fontsize=12)
    plt.ylabel(metric.upper() if metric != "mean_margin" else "Mean margin", fontsize=12)

    if metric in ("auroc", "acc"):
        plt.ylim(0.0, 1.02)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(alpha=0.25)

    # Put legend inside the figure
    plt.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[SAVE] {out_png}")

def plot_across_depth_all_models_two_tasks(curves_map, method: str, metric: str,
                                           out_png: str, normalize_depth=True):
    """
    One output image per (method, metric).
    Left subplot: SINGLE
    Right subplot: MCQ

    Each subplot contains one curve per model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    task_order = ["single", "mcq"]
    task_titles = {
        "single": "SINGLE",
        "mcq": "MCQ",
    }

    for ax, task in zip(axes, task_order):
        curves = curves_map.get((task, method), [])
        if not curves:
            continue

        for model_path, df in curves:
            df = normalize_df(df)
            last_layer = int(df["layer"].max())
            x = (df["layer"] / last_layer) if (normalize_depth and last_layer > 0) else df["layer"]

            ax.plot(
                x,
                df[metric],
                label=pretty_model(model_path),
                linewidth=2.2
            )

        ax.set_title(task_titles[task], fontsize=12)
        ax.set_xlabel(
            "Normalized depth (layer / last_layer)" if normalize_depth else "Layer",
            fontsize=11
        )

        if metric in ("auroc", "acc"):
            ax.set_ylim(0.0, 1.02)

        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=10)

        # legend inside each subplot
        ax.legend(loc="lower right", frameon=True, fontsize=9)

    if metric == "mean_margin":
        axes[0].set_ylabel("Mean margin", fontsize=11)
    else:
        axes[0].set_ylabel(metric.upper(), fontsize=11)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[SAVE] {out_png}")
def make_probe_family_table(curves_map, task: str, methods: list[str]) -> pd.DataFrame:
    """
    Table like your screenshot:
      Method | Mean_Peak_AUROC | Mean_Peak_ACC
    Mean is over models (for that task).
    """
    rows = []
    for method in methods:
        curves = curves_map.get((task, method), [])
        if not curves:
            continue

        peaks_auroc = []
        peaks_acc = []
        for _, df in curves:
            df = normalize_df(df)
            if df["auroc"].notna().any():
                peaks_auroc.append(float(df["auroc"].max()))
            if df["acc"].notna().any():
                peaks_acc.append(float(df["acc"].max()))

        rows.append({
            "Method": method.upper(),
            "Mean_Peak_AUROC": float(np.mean(peaks_auroc)) if peaks_auroc else np.nan,
            "Mean_Peak_ACC": float(np.mean(peaks_acc)) if peaks_acc else np.nan,
        })

    tbl = pd.DataFrame(rows).sort_values("Mean_Peak_AUROC", ascending=False)
    return tbl

def plot_probe_family_mean_metric_v1(curves_map, task: str, methods: list[str],
                                  metric: str, out_png: str, grid_n: int = 101):
    """
    Probe-family comparison plot (like your screenshot) for any metric:
      metric in {'auroc','acc','mean_margin'}

    X = normalized depth [0..1]
    Y = mean(metric) across models
    One curve per method.

    Each model is interpolated onto a common grid then averaged.
    """
    grid = np.linspace(0.0, 1.0, grid_n)
    plt.figure(figsize=(8.2, 5.8))

    for method in methods:
        curves = curves_map.get((task, method), [])
        if not curves:
            continue

        ys = []
        for _, df in curves:
            df = normalize_df(df)
            last_layer = int(df["layer"].max())
            if last_layer <= 0:
                continue

            x = df["layer"].to_numpy() / last_layer
            y = df[metric].to_numpy()

            # drop NaNs for safe interp
            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]
            y = y[m]
            if len(x) < 2:
                continue

            order = np.argsort(x)
            x = x[order]
            y = y[order]

            y_interp = np.interp(grid, x, y)
            ys.append(y_interp)

        if not ys:
            continue

        mean_y = np.mean(np.stack(ys, axis=0), axis=0)
        plt.plot(grid, mean_y, label=method.upper())

    plt.title(f"{task.upper()} – Probe family comparison (mean {metric.upper()} across depth)")
    plt.xlabel("Normalized depth (layer / last_layer)")
    if metric in ("auroc", "acc"):
        plt.ylabel(f"{metric.upper()} (mean across models)")
        plt.ylim(0.0, 1.02)
    else:
        plt.ylabel("Mean margin (mean across models)")
    plt.grid(alpha=0.25)
    #plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_png}")

def plot_probe_family_mean_metric(curves_map, task: str, methods: list[str],
                                  metric: str, out_png: str, grid_n: int = 101):
    """
    Probe-family comparison plot (like your screenshot) for any metric:
      metric in {'auroc','acc','mean_margin'}

    X = normalized depth [0..1]
    Y = mean(metric) across models
    One curve per method.

    Each model is interpolated onto a common grid then averaged.
    """
    grid = np.linspace(0.0, 1.0, grid_n)
    plt.figure(figsize=(8.2, 5.8))

    for method in methods:
        curves = curves_map.get((task, method), [])
        if not curves:
            continue

        ys = []
        for _, df in curves:
            df = normalize_df(df)
            last_layer = int(df["layer"].max())
            if last_layer <= 0:
                continue

            x = df["layer"].to_numpy() / last_layer
            y = df[metric].to_numpy()

            # drop NaNs for safe interp
            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]
            y = y[m]
            if len(x) < 2:
                continue

            order = np.argsort(x)
            x = x[order]
            y = y[order]

            y_interp = np.interp(grid, x, y)
            ys.append(y_interp)

        if not ys:
            continue

        mean_y = np.mean(np.stack(ys, axis=0), axis=0)
        plt.plot(grid, mean_y, label=method.upper(), linewidth=2.4)

    # Keep title minimal
    plt.title("", fontsize=12)
    plt.xlabel("Normalized depth (layer / last_layer)", fontsize=12)

    if metric in ("auroc", "acc"):
        plt.ylabel(f"{metric.upper()} (mean across models)", fontsize=12)
        plt.ylim(0.0, 1.02)
    else:
        plt.ylabel("Mean margin (mean across models)", fontsize=12)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(alpha=0.25)

    # Put legend inside the figure
    plt.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[SAVE] {out_png}")

def plot_probe_family_mean_metric_two_tasks(curves_map, methods: list[str],
                                            metric: str, out_png: str,
                                            grid_n: int = 101):
    """
    One output image per metric.
    Left subplot: SINGLE
    Right subplot: MCQ

    Example outputs:
      - mean_probe_family_auroc_two_tasks.png
      - mean_probe_family_acc_two_tasks.png
    """
    grid = np.linspace(0.0, 1.0, grid_n)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    task_order = ["single", "mcq"]
    task_titles = {
        "single": "SINGLE",
        "mcq": "MCQ",
    }

    for ax, task in zip(axes, task_order):
        for method in methods:
            curves = curves_map.get((task, method), [])
            if not curves:
                continue

            ys = []
            for _, df in curves:
                df = normalize_df(df)
                last_layer = int(df["layer"].max())
                if last_layer <= 0:
                    continue

                x = df["layer"].to_numpy() / last_layer
                y = df[metric].to_numpy()

                m = np.isfinite(x) & np.isfinite(y)
                x = x[m]
                y = y[m]
                if len(x) < 2:
                    continue

                order = np.argsort(x)
                x = x[order]
                y = y[order]

                y_interp = np.interp(grid, x, y)
                ys.append(y_interp)

            if not ys:
                continue

            mean_y = np.mean(np.stack(ys, axis=0), axis=0)
            ax.plot(grid, mean_y, label=method.upper(), linewidth=2.2)

        ax.set_title(task_titles[task], fontsize=12)
        ax.set_xlabel("Normalized depth (layer / last_layer)", fontsize=11)

        if metric in ("auroc", "acc"):
            ax.set_ylim(0.0, 1.02)

        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=10)

        # legend inside subplot
        ax.legend(loc="lower right", frameon=True, fontsize=9)

    if metric in ("auroc", "acc"):
        axes[0].set_ylabel(f"{metric.upper()} (mean across models)", fontsize=11)
    else:
        axes[0].set_ylabel("Mean margin (mean across models)", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[SAVE] {out_png}")
def main():
    ap = argparse.ArgumentParser("Probes: summary + plots + probe-family comparison (mcq/single)")
    ap.add_argument("--task", default="mcq,single")
    ap.add_argument("--models", 
                    help="comma list: EleutherAI__gpt-neo-125M,facebook__opt-125m,...", default="Qwen/Qwen2.5-0.5B,deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--out_root", default=str(REPO_ROOT / "out"))
    ap.add_argument("--methods", default="logreg,linsvm")

    # per-(task,method) plots (one curve per model)
    ap.add_argument("--plots", action="store_true",
                    help="Generate per-(task,method) curves across models (AUROC/ACC/Margin).")
    ap.add_argument("--plot_metrics", default="auroc,acc,mean_margin",
                    help="comma list among: auroc,acc,mean_margin")
    ap.add_argument("--normalize_depth", action="store_true",
                    help="Use normalized depth on x-axis (recommended).")

    # probe-family comparison outputs (one curve per method)
    ap.add_argument("--family_compare", action="store_true",
                    help="Generate probe-family table + mean metric-by-depth plot (one curve per method).")
    ap.add_argument("--family_grid_n", type=int, default=101,
                    help="Interpolation grid points for family mean plot.")
    ap.add_argument("--family_metrics", default="auroc,acc,mean_margin",
                    help="comma list among: auroc,acc,mean_margin (for family mean plot)")
    ap.add_argument("--edl_k", type=int, default=2)

    args = ap.parse_args()
    

    MODELS  = [m.strip() for m in args.models.split(",") if m.strip()]
    TASKS   = [t.strip() for t in args.task.split(",") if t.strip()]
    METHODS = [m.strip() for m in args.methods.split(",") if m.strip()]
    PLOT_METRICS = [m.strip() for m in args.plot_metrics.split(",") if m.strip()]
    FAMILY_METRICS = [m.strip() for m in args.family_metrics.split(",") if m.strip()]

    out_tables = os.path.join(args.out_root, "reports", "tables")
    out_figs   = os.path.join(args.out_root, "reports", "figures")
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    rows = []
    curves_map = {}  # key=(task, method) -> list of (model_path, df)

    # ---- Load all per-layer CSVs ----
    for model in MODELS:
        model_path = model.replace("/", "__")

        for task in TASKS:
            for method in METHODS:
                csv_path = read_metrics_csv(args.out_root, model_path, task, method)
                if not csv_path:
                    continue

                df = pd.read_csv(csv_path)
                df = normalize_df(df)

                # peaks
                best = {}
                for metric in ["acc", "auroc", "mean_margin"]:
                    if df[metric].notna().any():
                        idx = df[metric].idxmax()
                        best[metric] = (int(df.loc[idx, "layer"]), float(df.loc[idx, metric]))
                    else:
                        best[metric] = (None, np.nan)

                edl = edl_probe_margin_acc(df, task, k=args.edl_k)

                rows.append({
                    "model": model_path,
                    "task": task,
                    "method": method,
                    "best_layer_by_acc": best["acc"][0],
                    "acc_value": best["acc"][1],
                    "best_layer_by_auroc": best["auroc"][0],
                    "auroc_value": best["auroc"][1],
                    "best_layer_by_margin": best["mean_margin"][0],
                    "mean_margin": best["mean_margin"][1],
                    "EDL_probe": edl,
                    "csv_path": csv_path,
                })

                curves_map.setdefault((task, method), []).append((model_path, df))

    # ---- Summary CSV ----
    summary = pd.DataFrame(rows)
    out_csv = os.path.join(out_tables, "summary_all_models.csv")
    summary.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}")

    # ---- Per-method plots (one curve per model) ----
    if args.plots:
        for (task, method), curves in curves_map.items():
            for metric in PLOT_METRICS:
                title = f"{task.upper()} – {method.upper()} separability ({metric.upper()}) across depth"
                out_png = os.path.join(out_figs, f"{task}_{method}_{metric}_across_depth.png")
                plot_across_depth_all_models(
                    curves=curves,
                    metric=metric,
                    title=title,
                    out_png=out_png,
                    normalize_depth=args.normalize_depth
                )
        # combined two-task figures: one image per (method, metric)
        for method in METHODS:
            for metric in PLOT_METRICS:
                out_png = os.path.join(
                    out_figs,
                    f"{method}_{metric}_two_tasks.png"
                )

                plot_across_depth_all_models_two_tasks(
                    curves_map=curves_map,
                    method=method,
                    metric=metric,
                    out_png=out_png,
                    normalize_depth=True,
                )

    # ---- Probe-family compare (table + mean curves across models) ----
    if args.family_compare:
        for task in TASKS:
            # Table: mean peak AUROC/ACC across models
            tbl = make_probe_family_table(curves_map, task=task, methods=METHODS)
            out_tbl = os.path.join(out_tables, f"{task}_probe_family_comparison.csv")
            tbl.to_csv(out_tbl, index=False)
            print(f"[SAVE] {out_tbl}")

            # Family mean plots: for each requested metric (default: auroc,acc)
            for metric in FAMILY_METRICS:
                out_png = os.path.join(out_figs, f"{task}_probe_family_mean_{metric}_depth.png")
                plot_probe_family_mean_metric(
                    curves_map=curves_map,
                    task=task,
                    methods=METHODS,
                    metric=metric,
                    out_png=out_png,
                    grid_n=args.family_grid_n
                )

            # combined two-task figures: one image for AUROC, one image for ACC
            for metric in FAMILY_METRICS:
                if metric not in {"auroc", "acc"}:
                    continue

                out_png = os.path.join(
                    out_figs,
                    f"probe_family_mean_{metric}_two_tasks.png"
                )

                plot_probe_family_mean_metric_two_tasks(
                    curves_map=curves_map,
                    methods=METHODS,
                    metric=metric,
                    out_png=out_png,
                    grid_n=args.family_grid_n,
                )
if __name__ == "__main__":
    main()
