import argparse, os, glob
import pandas as pd
import numpy as np
from pathlib import Path
from textwrap import dedent

def load_method_csvs(tables_dir: str, task: str) -> dict[str, pd.DataFrame]:
    """
    Scan tables_dir/<task>/<method>/ for per-method CSVs produced by run_noise_injection.py
    returns: {method -> DataFrame}
    """
    root = Path(tables_dir)
    out = {}
    # accept both per-method subdir and flat files (fallback)
    per_method_dirs = sorted((root / task).glob("*")) if (root / task).exists() else []
    for d in per_method_dirs:
        if not d.is_dir(): 
            continue
        # expect: <task>_noise_robustness_<method>.csv
        files = list(d.glob(f"{task}_noise_robustness_*.csv"))
        for f in files:
            df = pd.read_csv(f)
            # infer method from filename
            m = f.stem.split(f"{task}_noise_robustness_")[-1]
            out[m] = df
    # flat fallback (if user didn't use per-method subdirs)
    if not out:
        for f in (root / task).glob(f"{task}_noise_robustness_*.csv"):
            df = pd.read_csv(f)
            m = f.stem.split(f"{task}_noise_robustness_")[-1]
            out[m] = df
    return out


def summarize_method(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    For one method DataFrame with columns: layer, sigma, acc, auroc, flip_rate
    Return table with columns: sigma, mean_delta_metric, mean_flip_rate
    """
    # baseline per layer
    base = df[df["sigma"] == 0.0].set_index("layer")[metric]
    rows = []
    for sig in sorted(df["sigma"].unique()):
        if sig == 0.0: 
            continue
        d = df[df["sigma"] == sig].set_index("layer")
        common = d.index.intersection(base.index)
        delta = d.loc[common, metric] - base.loc[common]
        mean_delta = float(delta.mean()) if not delta.empty else np.nan
        mean_flip  = float(d.loc[common, "flip_rate"].mean()) if not d.empty else np.nan
        rows.append({"sigma": sig, f"mean_delta_{metric}": mean_delta, "mean_flip_rate": mean_flip})
    return pd.DataFrame(rows)


def summarize_thresholds(thr_path: Path) -> pd.DataFrame:
    """
    Read robustness_thresholds_summary.csv and compute per-method aggregates of thresholds:
    mean & median of sigma_acc5, sigma_acc10, sigma_flip20 across layers.
    """
    if not thr_path.exists():
        return pd.DataFrame()
    thr = pd.read_csv(thr_path)
    # Helpful aggregates per method
    agg = thr.groupby("method").agg(
        sigma_acc5_mean  = ("sigma_acc5",  "mean"),
        sigma_acc5_median= ("sigma_acc5",  "median"),
        sigma_acc10_mean = ("sigma_acc10", "mean"),
        sigma_acc10_median=("sigma_acc10","median"),
        sigma_flip20_mean = ("sigma_flip20","mean"),
        sigma_flip20_median=("sigma_flip20","median"),
    ).reset_index()
    return agg


def main():
    ap = argparse.ArgumentParser("Summarize noise robustness CSVs into a compact report (per method).")
    ap.add_argument("--out_root", required=True, help="Root of out/ (same as run_noise_injection)")
    ap.add_argument("--model", required=True, help="HF model id (same string used in caching)")
    ap.add_argument("--task", choices=["mcq","single"], required=True)
    ap.add_argument("--metric", choices=["acc","auroc"], default="acc", help="delta metric to summarize")
    args = ap.parse_args()

    model_path = args.model.replace("/", "__")
    tables_dir = os.path.join(args.out_root, "reports", model_path, "tables", args.task)
    figures_dir = os.path.join(args.out_root, "reports", model_path, "figures", args.task)
    os.makedirs(figures_dir, exist_ok=True)

    # load per-method dataframes
    per_method = load_method_csvs(os.path.join(args.out_root, "reports", model_path, "tables"), args.task)
    if not per_method:
        raise FileNotFoundError(f"No per-method CSVs found under {tables_dir}")

    # thresholds summary
    thr_csv = Path(os.path.join(tables_dir, "robustness_thresholds_summary.csv"))
    thr_agg = summarize_thresholds(thr_csv)

    # build a flat summary table (method,sigma -> mean delta & flip)
    rows = []
    for m, df in per_method.items():
        part = summarize_method(df, metric=args.metric)
        part.insert(0, "method", m)
        rows.append(part)
    summary = pd.concat(rows, ignore_index=True).sort_values(["method","sigma"])

    # join thresholds (mean/median) per method
    if not thr_agg.empty:
        summary_agg = summary.groupby("method").agg(
            **{f"mean_delta_{args.metric}_avg": (f"mean_delta_{args.metric}", "mean"),
               "mean_flip_rate_avg": ("mean_flip_rate","mean")}
        ).reset_index()
        # merge both aggregates
        summary_methods = summary_agg.merge(thr_agg, on="method", how="left")
    else:
        summary_methods = summary.groupby("method").agg(
            **{f"mean_delta_{args.metric}_avg": (f"mean_delta_{args.metric}", "mean"),
               "mean_flip_rate_avg": ("mean_flip_rate","mean")}
        ).reset_index()

    # save CSVs
    out_csv_detail  = os.path.join(tables_dir, f"noise_summary_{args.metric}.csv")
    out_csv_methods = os.path.join(tables_dir, f"noise_summary_methods_{args.metric}.csv")
    summary.to_csv(out_csv_detail, index=False)
    summary_methods.to_csv(out_csv_methods, index=False)
    print("[SAVE]", out_csv_detail)
    print("[SAVE]", out_csv_methods)

    # produce a short Markdown narrative
    lines = []
    lines.append(f"# Robustness Summary — task={args.task}, metric={args.metric.upper()}")
    lines.append("")
    for _, row in summary_methods.sort_values("method").iterrows():
        m = row["method"]
        md_avg = row.get(f"mean_delta_{args.metric}_avg", np.nan)
        fr_avg = row.get("mean_flip_rate_avg", np.nan)
        sa5 = row.get("sigma_acc5_mean", np.nan)
        sa10 = row.get("sigma_acc10_mean", np.nan)
        sf20 = row.get("sigma_flip20_mean", np.nan)
        lines.append(f"## Method: {m}")
        lines.append(f"- Average Δ{args.metric.upper()} across sigmas: {md_avg:.4f}")
        lines.append(f"- Average flip-rate across sigmas: {fr_avg:.4f}")
        if not np.isnan(sa5):
            lines.append(f"- σ* (Δ{args.metric.upper()} ≤ -0.05) — mean: {sa5:.3f} | median: {row.get('sigma_acc5_median', np.nan):.3f}")
        if not np.isnan(sa10):
            lines.append(f"- σ* (Δ{args.metric.upper()} ≤ -0.10) — mean: {sa10:.3f} | median: {row.get('sigma_acc10_median', np.nan):.3f}")
        if not np.isnan(sf20):
            lines.append(f"- σ* (flip-rate ≥ 0.20) — mean: {sf20:.3f} | median: {row.get('sigma_flip20_median', np.nan):.3f}")
        lines.append("")

    md_text = "\n".join(lines)
    out_md = os.path.join(tables_dir, f"noise_summary_{args.metric}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_text)
    print("[SAVE]", out_md)

    # (Optional) print a short console preview
    print("\n=== Preview (methods aggregate) ===")
    print(summary_methods.to_string(index=False))

if __name__ == "__main__":
    main()
