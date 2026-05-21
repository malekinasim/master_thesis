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
    p1 = os.path.join(
        out_root,
        "reports",
        model_path,
        "lin_probs",
        "tables",
        task,
        f"{task}_{method}_perlayer_metrics_by_split.csv",
    )
    if os.path.exists(p1):
        return p1

    patt = os.path.join(
        out_root,
        "reports",
        model_path,
        "**",
        f"{task}_{method}_perlayer_metrics_by_split.csv",
    )
    hits = glob.glob(patt, recursive=True)
    return hits[0] if hits else None


def read_split_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["layer", "split", "acc", "auroc"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}. Found: {df.columns.tolist()}")

    df["layer"] = df["layer"].astype(int)
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["split"] = df["split"].replace({"val": "validation", "valid": "validation"})
    return df.sort_values(["split", "layer"]).reset_index(drop=True)


def metric_at_layer(df: pd.DataFrame, split: str, layer: int, metric: str) -> float:
    sub = df[(df["split"] == split) & (df["layer"] == layer)]
    if sub.empty or metric not in df.columns:
        return np.nan
    v = sub.iloc[0][metric]
    return float(v) if pd.notna(v) else np.nan


def format_float_df(df: pd.DataFrame, precision: int = 6) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=[np.floating]).columns
    for c in float_cols:
        out[c] = out[c].map(lambda x: f"{x:.{precision}f}" if pd.notna(x) else "")
    return out


# ---------------------------------------------------------------------
# Validation-anchored single-layer summary (l* = argmax validation AUROC)
# ---------------------------------------------------------------------
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

                val_df = df[df["split"] == "validation"].copy()
                if val_df.empty or val_df["auroc"].isna().all():
                    continue

                idx = val_df["auroc"].astype(float).idxmax()
                l_star = int(val_df.loc[idx, "layer"])

                train_auroc = metric_at_layer(df, "train", l_star, "auroc")
                val_auroc   = metric_at_layer(df, "validation", l_star, "auroc")
                test_auroc  = metric_at_layer(df, "test", l_star, "auroc")

                train_acc = metric_at_layer(df, "train", l_star, "acc")
                val_acc   = metric_at_layer(df, "validation", l_star, "acc")
                test_acc  = metric_at_layer(df, "test", l_star, "acc")

                train_mm = metric_at_layer(df, "train", l_star, "mean_margin")
                val_mm   = metric_at_layer(df, "validation", l_star, "mean_margin")
                test_mm  = metric_at_layer(df, "test", l_star, "mean_margin")

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

                    "train_mean_margin_at_lstar": train_mm,
                    "val_mean_margin_at_lstar": val_mm,
                    "test_mean_margin_at_lstar": test_mm,

                    "gap_auroc_train_minus_test": train_auroc - test_auroc,
                    "gap_acc_train_minus_test": train_acc - test_acc,
                    "gap_mean_margin_train_minus_test": (
                        train_mm - test_mm
                        if pd.notna(train_mm) and pd.notna(test_mm)
                        else np.nan
                    ),
                })

    return pd.DataFrame(rows)


def summarise_val_anchored(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    tmp = df.copy()
    tmp["abs_gap_auroc"] = tmp["gap_auroc_train_minus_test"].abs()
    tmp["abs_gap_acc"] = tmp["gap_acc_train_minus_test"].abs()
    tmp["abs_gap_mean_margin"] = tmp["gap_mean_margin_train_minus_test"].abs()

    return (
        tmp.groupby(["task", "method"], as_index=False)
        .agg(
            n_models=("model_path", "count"),

            mean_gap_auroc=("gap_auroc_train_minus_test", "mean"),
            median_gap_auroc=("gap_auroc_train_minus_test", "median"),
            mean_abs_gap_auroc=("abs_gap_auroc", "mean"),
            max_abs_gap_auroc=("abs_gap_auroc", "max"),

            mean_gap_acc=("gap_acc_train_minus_test", "mean"),
            median_gap_acc=("gap_acc_train_minus_test", "median"),
            mean_abs_gap_acc=("abs_gap_acc", "mean"),
            max_abs_gap_acc=("abs_gap_acc", "max"),

            mean_gap_mean_margin=("gap_mean_margin_train_minus_test", "mean"),
            median_gap_mean_margin=("gap_mean_margin_train_minus_test", "median"),
            mean_abs_gap_mean_margin=("abs_gap_mean_margin", "mean"),
            max_abs_gap_mean_margin=("abs_gap_mean_margin", "max"),
        )
        .sort_values(["task", "method"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Across-layer gap summary
# ---------------------------------------------------------------------
def compute_all_layer_gap_rows(out_root: str, models: list[str], tasks: list[str], methods: list[str]) -> pd.DataFrame:
    rows = []

    for model in models:
        model_path = model.replace("/", "__")

        for task in tasks:
            for method in methods:
                csv_path = find_by_split_csv(out_root, model_path, task, method)
                if not csv_path:
                    continue

                df = read_split_df(csv_path)
                layers = sorted(df["layer"].unique())

                for li in layers:
                    train_auroc = metric_at_layer(df, "train", li, "auroc")
                    test_auroc  = metric_at_layer(df, "test", li, "auroc")

                    train_acc = metric_at_layer(df, "train", li, "acc")
                    test_acc  = metric_at_layer(df, "test", li, "acc")

                    train_mm = metric_at_layer(df, "train", li, "mean_margin")
                    test_mm  = metric_at_layer(df, "test", li, "mean_margin")

                    if not np.isfinite(train_auroc) or not np.isfinite(test_auroc):
                        continue

                    rows.append({
                        "model_path": model_path,
                        "model_label": pretty_model(model_path),
                        "task": task,
                        "method": method.upper(),
                        "layer": int(li),

                        "train_auroc": train_auroc,
                        "test_auroc": test_auroc,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "train_mean_margin": train_mm,
                        "test_mean_margin": test_mm,

                        "gap_auroc_train_minus_test": train_auroc - test_auroc,
                        "gap_acc_train_minus_test": train_acc - test_acc,
                        "gap_mean_margin_train_minus_test": (
                            train_mm - test_mm
                            if pd.notna(train_mm) and pd.notna(test_mm)
                            else np.nan
                        ),
                    })

    return pd.DataFrame(rows)


def summarise_all_layers_per_model(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all.copy()

    tmp = df_all.copy()
    tmp["abs_gap_auroc"] = tmp["gap_auroc_train_minus_test"].abs()
    tmp["abs_gap_acc"] = tmp["gap_acc_train_minus_test"].abs()
    tmp["abs_gap_mean_margin"] = tmp["gap_mean_margin_train_minus_test"].abs()

    return (
        tmp.groupby(["model_path", "model_label", "task", "method"], as_index=False)
        .agg(
            n_layers=("layer", "count"),

            mean_gap_auroc=("gap_auroc_train_minus_test", "mean"),
            median_gap_auroc=("gap_auroc_train_minus_test", "median"),
            mean_abs_gap_auroc=("abs_gap_auroc", "mean"),
            max_abs_gap_auroc=("abs_gap_auroc", "max"),

            mean_gap_acc=("gap_acc_train_minus_test", "mean"),
            median_gap_acc=("gap_acc_train_minus_test", "median"),
            mean_abs_gap_acc=("abs_gap_acc", "mean"),
            max_abs_gap_acc=("abs_gap_acc", "max"),

            mean_gap_mean_margin=("gap_mean_margin_train_minus_test", "mean"),
            median_gap_mean_margin=("gap_mean_margin_train_minus_test", "median"),
            mean_abs_gap_mean_margin=("abs_gap_mean_margin", "mean"),
            max_abs_gap_mean_margin=("abs_gap_mean_margin", "max"),
        )
        .sort_values(["task", "method", "model_label"])
        .reset_index(drop=True)
    )


def summarise_all_layers_by_task_method(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all.copy()

    tmp = df_all.copy()
    tmp["abs_gap_auroc"] = tmp["gap_auroc_train_minus_test"].abs()
    tmp["abs_gap_acc"] = tmp["gap_acc_train_minus_test"].abs()
    tmp["abs_gap_mean_margin"] = tmp["gap_mean_margin_train_minus_test"].abs()

    return (
        tmp.groupby(["task", "method"], as_index=False)
        .agg(
            n_rows=("layer", "count"),
            n_models=("model_path", "nunique"),

            mean_gap_auroc=("gap_auroc_train_minus_test", "mean"),
            median_gap_auroc=("gap_auroc_train_minus_test", "median"),
            mean_abs_gap_auroc=("abs_gap_auroc", "mean"),
            max_abs_gap_auroc=("abs_gap_auroc", "max"),

            mean_gap_acc=("gap_acc_train_minus_test", "mean"),
            median_gap_acc=("gap_acc_train_minus_test", "median"),
            mean_abs_gap_acc=("abs_gap_acc", "mean"),
            max_abs_gap_acc=("abs_gap_acc", "max"),

            mean_gap_mean_margin=("gap_mean_margin_train_minus_test", "mean"),
            median_gap_mean_margin=("gap_mean_margin_train_minus_test", "median"),
            mean_abs_gap_mean_margin=("abs_gap_mean_margin", "mean"),
            max_abs_gap_mean_margin=("abs_gap_mean_margin", "max"),
        )
        .sort_values(["task", "method"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
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
    plt.boxplot(data, tick_labels=labels, showmeans=True)
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
        sub["test_auroc_at_lstar"].values,
    ]).astype(float)
    allv = allv[np.isfinite(allv)]

    if len(allv) > 0:
        lo, hi = float(np.min(allv)), float(np.max(allv))
        plt.plot([lo, hi], [lo, hi], linewidth=1)

    plt.xlabel("Full-train AUROC at l* (validation-anchored)")
    plt.ylabel("Test AUROC at l* (validation-anchored)")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_png}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        "Generalisation gap for probes: validation-anchored summary + across-layer summary"
    )
    ap.add_argument("--out_root", default="out")
    ap.add_argument(
        "--models",
        required=True,
        help="Comma-separated HF names or model_path with __ "
             "(e.g. EleutherAI__gpt-neo-125M,Qwen__Qwen2.5-0.5B,...)"
    )
    ap.add_argument("--tasks", default="mcq,single")
    ap.add_argument("--methods", default="massmean,lda,logreg,linsvm")
    ap.add_argument("--precision", type=int, default=6)
    args = ap.parse_args()

    out_root = args.out_root
    out_tables = os.path.join(out_root, "reports", "tables")
    out_figs = os.path.join(out_root, "reports", "figures")
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # ---------------------------
    # 1) Validation-anchored l*
    # ---------------------------
    df_val = compute_gap_rows(out_root, models, tasks, methods)
    if df_val.empty:
        raise SystemExit("No rows produced. Ensure per-layer CSVs include split=validation.")

    out_csv_val_raw = os.path.join(out_tables, "table_gap_per_model_val_anchored_raw.csv")
    df_val.to_csv(out_csv_val_raw, index=False)
    print(f"[SAVE] {out_csv_val_raw}")

    out_csv_val_fmt = os.path.join(out_tables, "table_gap_per_model_val_anchored.csv")
    format_float_df(df_val, precision=args.precision).to_csv(out_csv_val_fmt, index=False)
    print(f"[SAVE] {out_csv_val_fmt}")

    summ_val = summarise_val_anchored(df_val)

    out_csv_val_summ_raw = os.path.join(out_tables, "table_gap_summary_val_anchored_raw.csv")
    summ_val.to_csv(out_csv_val_summ_raw, index=False)
    print(f"[SAVE] {out_csv_val_summ_raw}")

    out_csv_val_summ_fmt = os.path.join(out_tables, "table_gap_summary_val_anchored.csv")
    format_float_df(summ_val, precision=args.precision).to_csv(out_csv_val_summ_fmt, index=False)
    print(f"[SAVE] {out_csv_val_summ_fmt}")

    # ---------------------------
    # 2) Across all layers
    # ---------------------------
    df_all = compute_all_layer_gap_rows(out_root, models, tasks, methods)
    if not df_all.empty:
        out_csv_all_raw = os.path.join(out_tables, "table_gap_all_layers_raw.csv")
        df_all.to_csv(out_csv_all_raw, index=False)
        print(f"[SAVE] {out_csv_all_raw}")

        out_csv_all_fmt = os.path.join(out_tables, "table_gap_all_layers.csv")
        format_float_df(df_all, precision=args.precision).to_csv(out_csv_all_fmt, index=False)
        print(f"[SAVE] {out_csv_all_fmt}")

        summ_all_per_model = summarise_all_layers_per_model(df_all)
        out_csv_all_per_model_raw = os.path.join(
            out_tables,
            "table_gap_per_model_all_layers_summary_raw.csv"
        )
        summ_all_per_model.to_csv(out_csv_all_per_model_raw, index=False)
        print(f"[SAVE] {out_csv_all_per_model_raw}")

        out_csv_all_per_model_fmt = os.path.join(
            out_tables,
            "table_gap_per_model_all_layers_summary.csv"
        )
        format_float_df(summ_all_per_model, precision=args.precision).to_csv(
            out_csv_all_per_model_fmt, index=False
        )
        print(f"[SAVE] {out_csv_all_per_model_fmt}")

        summ_all_task_method = summarise_all_layers_by_task_method(df_all)
        out_csv_all_task_method_raw = os.path.join(
            out_tables,
            "table_gap_summary_all_layers_raw.csv"
        )
        summ_all_task_method.to_csv(out_csv_all_task_method_raw, index=False)
        print(f"[SAVE] {out_csv_all_task_method_raw}")

        out_csv_all_task_method_fmt = os.path.join(
            out_tables,
            "table_gap_summary_all_layers.csv"
        )
        format_float_df(summ_all_task_method, precision=args.precision).to_csv(
            out_csv_all_task_method_fmt, index=False
        )
        print(f"[SAVE] {out_csv_all_task_method_fmt}")

    # ---------------------------
    # 3) Plots from validation-anchored table
    # ---------------------------
    for task in tasks:
        plot_gap_box(
            df_val,
            task,
            metric_col="gap_auroc_train_minus_test",
            out_png=os.path.join(out_figs, f"fig_B5_gap_box_auroc_train_test_{task}.png"),
            title=(
                f"{task.upper()} – Final probe generalisation gap across models "
                f"(ΔAUROC = full-train − test at l*, validation-anchored)"
            ),
        )

        plot_train_vs_test_scatter(
            df_val,
            task,
            out_png=os.path.join(out_figs, f"fig_B5_train_vs_test_scatter_{task}.png"),
            title=(
                f"{task.upper()} – Full-train vs Test AUROC at l* "
                f"(validation-anchored, by probe family)"
            ),
        )


if __name__ == "__main__":
    main()