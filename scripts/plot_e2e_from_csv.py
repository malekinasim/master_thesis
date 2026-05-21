#!/usr/bin/env python3
"""
Plot E2E robustness results from saved CSVs (no model forward).
Reads:
 - {task}_e2e_noise_curves.csv   (per layer/sigma ACC)
 - {task}_e2e_robustness_per_layer.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_per_layer_curves(df, figs_dir):
    layers = sorted(df["layer"].unique())
    for li in layers:
        sub = df[df["layer"] == li].sort_values("sigma")
        plt.figure(figsize=(5, 4))
        plt.plot(sub["sigma"], sub["acc"], marker="o")
        plt.xlabel("sigma (noise std)")
        plt.ylabel("Accuracy")
        plt.title(f"Layer {li}: ACC vs sigma")
        plt.grid(alpha=0.3)
        out_png = figs_dir / f"acc_vs_sigma_layer{li}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()
        print("[SAVE]", out_png)


def plot_robustness(df, figs_dir, task, model):
    plt.figure(figsize=(7, 4))
    plt.plot(df["layer"], df["robustness_auc"], marker="s")
    plt.xlabel("Layer")
    plt.ylabel("Robustness (AUC of ACC vs sigma)")
    plt.title(f"{task.upper()} robustness per layer ({model})")
    plt.grid(alpha=0.3)
    out_png = figs_dir / f"{task}_robustness_vs_layer.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print("[SAVE]", out_png)


def main():
    ap = argparse.ArgumentParser("Plot E2E robustness from CSVs")
    ap.add_argument("--task", choices=["mcq", "single"], required=True)
    ap.add_argument("--model", required=True, help="model name used in reports (e.g., EleutherAI/gpt-neo-125M)")
    ap.add_argument("--out_root", default=str(Path(__file__).resolve().parents[1] / "out"))
    args = ap.parse_args()

    model_path = args.model.replace("/", "__")

    tables_dir = Path(args.out_root) / "reports" / model_path /"robustness"/"e2e"/"tables"/args.task 
    figs_dir = Path(args.out_root) / "reports" / model_path /"robustness"/"e2e"/"figures"/args.task 
    figs_dir.mkdir(parents=True, exist_ok=True)

    curves_csv = tables_dir / f"{args.task}_e2e_noise_curves.csv"
    rob_csv = tables_dir / f"{args.task}_e2e_robustness_per_layer.csv"

    if curves_csv.exists():
        df_curves = pd.read_csv(curves_csv)
        plot_per_layer_curves(df_curves, figs_dir)
    else:
        print("[WARN] curves CSV not found:", curves_csv)

    if rob_csv.exists():
        df_rob = pd.read_csv(rob_csv)
        plot_robustness(df_rob, figs_dir, args.task, args.model)
    else:
        print("[WARN] robustness CSV not found:", rob_csv)


if __name__ == "__main__":
    main()
