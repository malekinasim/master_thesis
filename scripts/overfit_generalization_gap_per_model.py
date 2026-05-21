import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LABELS = {
    "EleutherAI__gpt-neo-125M": "GPT-Neo-125M",
    "Qwen__Qwen2.5-0.5B": "Qwen2.5-0.5B",
    "TinyLlama__TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
    "deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-R1-Distill (1.5B)",
    "facebook__opt-125m": "OPT-125M",
    "meta-llama__Llama-3.2-1B": "Llama-3.2-1B",
}


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pretty_model(model_path: str) -> str:
    return LABELS.get(model_path, model_path.replace("__", "/"))


def find_by_split_csv(out_root: str, model_path: str, task: str, method: str) -> str | None:
    p1 = os.path.join(out_root, "reports", model_path, "lin_probs", "tables", task,
                      f"{task}_{method}_perlayer_metrics_by_split.csv")
    if os.path.exists(p1):
        return p1
    patt = os.path.join(out_root, "reports", model_path, "**", f"{task}_{method}_perlayer_metrics_by_split.csv")
    hits = glob.glob(patt, recursive=True)
    return hits[0] if hits else None


def read_split_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["layer", "split", "acc", "auroc"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}. Found: {df.columns.tolist()}")
    df["layer"] = df["layer"].astype(int)
    df["split"] = df["split"].astype(str).str.lower()
    df["split"] = df["split"].replace({"val": "validation", "valid": "validation"})
    return df.sort_values(["split", "layer"])


def metric_at_layer(df: pd.DataFrame, split: str, layer: int, metric: str) -> float:
    sub = df[(df["split"] == split) & (df["layer"] == layer)]
    if sub.empty or metric not in df.columns:
        return np.nan
    v = sub.iloc[0][metric]
    return float(v) if pd.notna(v) else np.nan


def compute_gap_rows(out_root: str, models: list[str], tasks: list[str], methods: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        model_path = model.replace("/", "__")

        for task in tasks:
            for method in methods:
                csv_path = find_by_split_csv(out_root, model_path, task, method)
                if not csv_path:
                    continue

                df = read_split_df(csv_path)

                # Anchor layer on VALIDATION AUROC
                val_df = df[df["split"] == "validation"].copy()
                if val_df.empty or val_df["auroc"].isna().all():
                    continue

                idx = val_df["auroc"].astype(float).idxmax()
                l_star = int(val_df.loc[idx, "layer"])

                train_auroc = metric_at_layer(df, "train", l_star, "auroc")
                test_auroc  = metric_at_layer(df, "test", l_star, "auroc")
                train_acc   = metric_at_layer(df, "train", l_star, "acc")
                test_acc    = metric_at_layer(df, "test", l_star, "acc")
                val_auroc   = metric_at_layer(df, "validation", l_star, "auroc")
                val_acc     = metric_at_layer(df, "validation", l_star, "acc")

                if not np.isfinite(train_auroc) or not np.isfinite(test_auroc):
                    continue

                rows.append({
                    "model_path": model_path,
                    "model_label": pretty_model(model_path),
                    "task": task,
                    "method": method.upper(),

                    "peak_layer_by_val_auroc": l_star,
                    "train_auroc_at_lstar": train_auroc,
                    "val_auroc_at_lstar": val_auroc,
                    "test_auroc_at_lstar": test_auroc,

                    "train_acc_at_lstar": train_acc,
                    "val_acc_at_lstar": val_acc,
                    "test_acc_at_lstar": test_acc,

                    "gap_auroc_train_minus_test": train_auroc - test_auroc,
                    "gap_acc_train_minus_test": train_acc - test_acc,
                })

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["task", "method"], as_index=False).agg(
        n_models=("model_path", "count"),
        mean_gap_auroc=("gap_auroc_train_minus_test", "mean"),
        mean_gap_acc=("gap_acc_train_minus_test", "mean"),
        median_gap_auroc=("gap_auroc_train_minus_test", "median"),
        median_gap_acc=("gap_acc_train_minus_test", "median"),
    ).sort_values(["task", "method"])


def plot_gap_box(df_gap: pd.DataFrame, task: str, metric_col: str, out_png: str, title: str):
    sub = df_gap[df_gap["task"] == task].copy()
    if sub.empty:
        return

    methods = sorted(sub["method"].unique())
    data, labels = [], []
    for m in methods:
        vals = sub[sub["method"] == m][metric_col].dropna().astype(float).values
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(m)

    if not data:
        return

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel(metric_col.replace("_", " "))
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_png}")


def plot_train_vs_test_scatter(df_gap: pd.DataFrame, task: str, out_png: str, title: str):
    sub = df_gap[df_gap["task"] == task].copy()
    if sub.empty:
        return

    plt.figure(figsize=(7.5, 6))
    for m in sorted(sub["method"].unique()):
        ss = sub[sub["method"] == m]
        x = ss["train_auroc_at_lstar"].astype(float).values
        y = ss["test_auroc_at_lstar"].astype(float).values
        plt.scatter(x, y, label=m, alpha=0.9)

    allv = np.concatenate([
        sub["train_auroc_at_lstar"].values,
        sub["test_auroc_at_lstar"].values
    ]).astype(float)
    allv = allv[np.isfinite(allv)]
    if len(allv) > 0:
        lo, hi = float(np.min(allv)), float(np.max(allv))
        plt.plot([lo, hi], [lo, hi], linewidth=1)  # y=x line

    plt.xlabel("Train AUROC at l* (validation-anchored)")
    plt.ylabel("Test AUROC at l* (validation-anchored)")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_png}")


def main():
    ap = argparse.ArgumentParser("Validation-anchored generalisation gap for probes (with plots)")
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--models", required=True,
                    help="Comma-separated HF names or model_path with __ (e.g. EleutherAI__gpt-neo-125M,...)")
    ap.add_argument("--tasks", default="mcq,single")
    ap.add_argument("--methods", default="massmean,lda,logreg,linsvm")
    args = ap.parse_args()

    out_root = args.out_root
    out_tables = os.path.join(out_root, "reports", "tables")
    out_figs = os.path.join(out_root, "reports", "figures")
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    df = compute_gap_rows(out_root, models, tasks, methods)
    if df.empty:
        raise SystemExit("No rows produced. Ensure per-layer CSVs include split=validation.")

    out_csv = os.path.join(out_tables, "table_gap_per_model_val_anchored.csv")
    df.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}")

    summ = summarise(df)
    out_csv2 = os.path.join(out_tables, "table_gap_summary_val_anchored.csv")
    summ.to_csv(out_csv2, index=False)
    print(f"[SAVE] {out_csv2}")

    # IMPORTANT: plots must use df (per-model), not summ (summary)
    for task in tasks:
        plot_gap_box(
            df, task,
            metric_col="gap_auroc_train_minus_test",
            out_png=os.path.join(out_figs, f"fig_B5_gap_box_auroc_train_test_{task}.png"),
            title=f"{task.upper()} – Generalisation gap across models (ΔAUROC = train − test at l*, validation-anchored)"
        )

        plot_train_vs_test_scatter(
            df, task,
            out_png=os.path.join(out_figs, f"fig_B5_train_vs_test_scatter_{task}.png"),
            title=f"{task.upper()} – Train vs Test AUROC at l* (validation-anchored, by probe family)"
        )


if __name__ == "__main__":
    main()