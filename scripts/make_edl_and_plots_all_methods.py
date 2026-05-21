import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# Labels / Pretty names
# -----------------------
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

def hf_name(model_path: str) -> str:
    return model_path.replace("__", "/")


# -----------------------
# IO helpers
# -----------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "layer" not in df.columns:
        raise ValueError(f"Missing 'layer' column. Columns={df.columns.tolist()}")
    df["layer"] = df["layer"].astype(int)

    # ensure these exist for all methods
    for c in ["acc", "auroc", "mean_margin"]:
        if c not in df.columns:
            df[c] = np.nan

    return df.sort_values("layer")

def find_metrics_csv(out_root: str, model_path: str, task: str, method: str) -> str | None:
    """
    Finds per-layer metrics CSV:
      out/reports/<model>/tables/<task>/<task>_<method>_perlayer_metrics.csv
    OR out/reports/<model>/lin_probs/tables/<task>/<task>_<method>_perlayer_metrics.csv
    OR recursively anywhere under that model folder.
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


# -----------------------
# EDL by AUROC
# -----------------------
def edl_start_end_by_auroc(df: pd.DataFrame, thr: float = 0.9, k: int = 2):
    """
    EDL start: earliest layer where AUROC >= thr for k consecutive layers
    EDL end  : last layer in the same contiguous above-threshold run
               that begins at EDL start
    """
    df = normalize_df(df)
    ok = (df["auroc"].fillna(-np.inf) >= thr).to_numpy()

    if len(ok) < k:
        return None, None

    roll = pd.Series(ok).rolling(k).apply(lambda s: s.all(), raw=True).to_numpy()
    ends = np.where(roll == 1)[0]
    if len(ends) == 0:
        return None, None

    start_idx = int(ends[0] - k + 1)
    edl_start = int(df.iloc[start_idx]["layer"])

    end_idx = start_idx
    while end_idx + 1 < len(ok) and ok[end_idx + 1]:
        end_idx += 1

    edl_end = int(df.iloc[end_idx]["layer"])
    return edl_start, edl_end

def make_edl_table(out_root: str, models: list[str], task: str, method: str, thr: float = 0.9, k: int = 2,
                   round_decimals: int = 3) -> pd.DataFrame:
    """
    Table like your Table 6 but for any (task, method).
    Columns:
      Model, Layers, EDL start, EDL end, Peak AUROC, Peak layer, ACC @ peak AUROC
    """
    rows = []
    for model_path in models:
        csv_path = find_metrics_csv(out_root, model_path, task=task, method=method)
        if not csv_path:
            continue

        df = normalize_df(pd.read_csv(csv_path))
        layers = int(df["layer"].max()) + 1

        edl_start, edl_end = edl_start_end_by_auroc(df, thr=thr, k=k)

        if df["auroc"].notna().any():
            idx_peak = df["auroc"].idxmax()
            peak_auroc = float(df.loc[idx_peak, "auroc"])
            peak_layer = int(df.loc[idx_peak, "layer"])
            acc_at_peak = float(df.loc[idx_peak, "acc"]) if pd.notna(df.loc[idx_peak, "acc"]) else np.nan
        else:
            peak_auroc, peak_layer, acc_at_peak = np.nan, None, np.nan

        rows.append({
            "Task": task,
            "Method": method.upper(),
            "Model": hf_name(model_path),
            "Layers": layers,
            f"EDL start (AUROC≥{thr}, K={k})": "N/A" if edl_start is None else int(edl_start),
            f"EDL end (AUROC≥{thr})": "N/A" if edl_end is None else int(edl_end),
            "Peak AUROC": round(peak_auroc, round_decimals) if pd.notna(peak_auroc) else np.nan,
            "Peak layer": peak_layer if peak_layer is not None else "N/A",
            "ACC @ peak AUROC": round(acc_at_peak, round_decimals) if pd.notna(acc_at_peak) else np.nan,
        })

    return pd.DataFrame(rows)


# -----------------------
# Plots: one curve per model
# -----------------------
def plot_task_method_metric(out_root: str, models: list[str], task: str, method: str,
                            metric: str, out_png: str, title: str,
                            normalize_depth: bool = True):
    """
    One curve per model for fixed (task, method, metric).
    """
    plt.figure(figsize=(9, 6))
    plotted = 0

    for model_path in models:
        csv_path = find_metrics_csv(out_root, model_path, task=task, method=method)
        if not csv_path:
            continue

        df = normalize_df(pd.read_csv(csv_path))

        last_layer = int(df["layer"].max())
        if normalize_depth and last_layer > 0:
            x = df["layer"] / last_layer
        else:
            x = df["layer"]

        if metric not in df.columns:
            continue

        plt.plot(x, df[metric], label=pretty_model(model_path))
        plotted += 1

    if plotted == 0:
        plt.close()
        print(f"[SKIP] No data to plot: task={task}, method={method}, metric={metric}")
        return

    plt.title(title)
    plt.xlabel("Normalized depth (layer / last_layer)" if normalize_depth else "Layer")
    plt.ylabel(metric.upper() if metric != "mean_margin" else "Mean margin")

    if metric in ("auroc", "acc"):
        plt.ylim(0.0, 1.02)

    plt.grid(alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_png}")


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser("Generate EDL tables + AUROC/ACC plots for ALL methods (mcq/single)")
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--models", required=True,
                    help="comma list: EleutherAI__gpt-neo-125M,facebook__opt-125m,...")
    ap.add_argument("--tasks", default="mcq",
                    help="comma list: mcq,single")
    ap.add_argument("--methods", default="logreg,lda,linsvm,massmean",
                    help="comma list: logreg,lda,linsvm,massmean")
    ap.add_argument("--metrics", default="auroc,acc",
                    help="comma list: auroc,acc,mean_margin")
    ap.add_argument("--normalize_depth", action="store_true",
                    help="use normalized depth on x-axis")

    # EDL table
    ap.add_argument("--make_tables", action="store_true",
                    help="generate EDL table for each (task, method) and a combined table")
    ap.add_argument("--edl_thr", type=float, default=0.9)
    ap.add_argument("--edl_k", type=int, default=2)
    ap.add_argument("--round", type=int, default=3)

    args = ap.parse_args()

    models  = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks   = [t.strip() for t in args.tasks.split(",") if t.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    out_tables = os.path.join(args.out_root, "reports", "tables")
    out_figs   = os.path.join(args.out_root, "reports", "figures")
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    combined_tables = []

    for task in tasks:
        for method in methods:
            # ---- plots for each metric ----
            for metric in metrics:
                out_png = os.path.join(out_figs, f"fig_{task}_{method}_{metric}_depth.png")
                title = f"{task.upper()} – {method.upper()} {metric.upper()} across depth"
                plot_task_method_metric(
                    out_root=args.out_root,
                    models=models,
                    task=task,
                    method=method,
                    metric=metric,
                    out_png=out_png,
                    title=title,
                    normalize_depth=args.normalize_depth
                )

            # ---- per-method EDL table ----
            if args.make_tables:
                tbl = make_edl_table(
                    out_root=args.out_root,
                    models=models,
                    task=task,
                    method=method,
                    thr=args.edl_thr,
                    k=args.edl_k,
                    round_decimals=args.round
                )
                if not tbl.empty:
                    
                    thr_tag = str(args.edl_thr).replace(".", "p")
                    out_csv = os.path.join(out_tables, f"table_edl_{task}_{method}_thr{thr_tag}_k{args.edl_k}.csv")
                    tbl.to_csv(out_csv, index=False)
                    print(f"[SAVE] {out_csv}")
                    combined_tables.append(tbl)

    # ---- combined table for all task/method ----
    if args.make_tables and combined_tables:
        combo = pd.concat(combined_tables, ignore_index=True)
        thr_tag = str(args.edl_thr).replace(".", "p")
        out_csv = os.path.join(out_tables, f"table_edl_all_tasks_all_methods_thr{thr_tag}_k{args.edl_k}.csv")
        combo.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")


if __name__ == "__main__":
    main()
