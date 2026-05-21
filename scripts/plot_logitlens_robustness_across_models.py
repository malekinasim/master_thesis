#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def sanitize_model_name(model_dir_name: str) -> str:
    return model_dir_name.replace("__", "/")


def pretty_model_name(model_dir_name: str) -> str:
    raw = sanitize_model_name(model_dir_name)
    replacements = {
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-R1-Distill (1.5B)",
        "Qwen/Qwen2.5-0.5B": "Qwen2.5 (0.5B)",
        "Qwen/Qwen2.5-1.5B": "Qwen2.5 (1.5B)",
    }
    return replacements.get(raw, raw)


def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
    })


def find_robustness_csvs(out_root: Path, task: str):
    pattern = f"reports/*/robustness/local/logitlens_local/{task}/tables/*_local_logitlens_robustness_per_layer.csv"
    return sorted(out_root.glob(pattern))


def find_noise_curve_csvs(out_root: Path, task: str):
    pattern = f"reports/*/robustness/local/logitlens_local/{task}/tables/*_local_logitlens_noise_curves.csv"
    return sorted(out_root.glob(pattern))


def plot_robustness_vs_depth(out_root: Path, task: str, save_dir: Path):
    apply_plot_style()

    csvs = find_robustness_csvs(out_root, task)
    if not csvs:
        raise FileNotFoundError(f"No robustness_per_layer CSVs found for task={task}")

    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    for csv_path in csvs:
        model_dir = csv_path.parts[csv_path.parts.index("reports") + 1]
        model_name = pretty_model_name(model_dir)

        df = pd.read_csv(csv_path)
        required = {"layer", "robustness_auc"}
        if not required.issubset(df.columns):
            print(f"[WARN] Skipping {csv_path} because required columns are missing")
            continue

        df = df.sort_values("layer").reset_index(drop=True)
        last_layer = max(int(df["layer"].max()), 1)
        df["norm_depth"] = df["layer"] / last_layer

        ax.plot(
            df["norm_depth"],
            df["robustness_auc"],
            linewidth=2.4,
            marker="o",
            markersize=4.5,
            label=model_name,
        )

    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Robustness AUC")
    ax.set_title(f"Layer-wise local logit-lens robustness ({task.upper()})")
    ax.grid(True, alpha=0.25)

    # legend inside figure
    ax.legend(loc="best", frameon=True)

    ax.margins(x=0.02, y=0.08)

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / f"{task}_local_logitlens_robustness_vs_depth_all_models.png"
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_png}")


def plot_noise_curves_for_one_model(
    out_root: Path,
    task: str,
    model_dir_name: str,
    save_dir: Path,
    layers=None,
):
    apply_plot_style()

    csv_path = (
        out_root
        / "reports"
        / model_dir_name
        / "robustness"
        / "local"
        / "logitlens_local"
        / task
        / "tables"
        / f"{task}_local_logitlens_noise_curves.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"Noise curve CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"layer", "sigma", "acc"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} is missing required columns: {required}")

    df = df.sort_values(["layer", "sigma"]).reset_index(drop=True)

    if layers is None or len(layers) == 0:
        all_layers = sorted(df["layer"].unique())
        if len(all_layers) <= 6:
            chosen = all_layers
        else:
            chosen = sorted(set([
                all_layers[0],
                all_layers[len(all_layers) // 4],
                all_layers[(2 * len(all_layers)) // 4],
                all_layers[(3 * len(all_layers)) // 4],
                all_layers[-1],
            ]))
    else:
        chosen = layers

    model_name = pretty_model_name(model_dir_name)

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for layer in chosen:
        g = df[df["layer"] == layer].sort_values("sigma")
        if not g.empty:
            ax.plot(
                g["sigma"],
                g["acc"],
                marker="o",
                markersize=4,
                linewidth=2.1,
                label=f"Layer {layer}",
            )

    ax.set_xlabel("Noise level ($\\sigma$)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Local logit-lens noise curves ({task.upper()})\n{model_name}")
    ax.grid(True, alpha=0.25)

    # legend inside figure
    ax.legend(loc="best", frameon=True)

    ax.margins(x=0.02, y=0.08)

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / f"{task}_local_logitlens_noise_curves_{model_dir_name}.png"
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_png}")


def plot_robustness_per_layer_for_one_model(
    out_root: Path,
    task: str,
    model_dir_name: str,
    save_dir: Path,
):
    apply_plot_style()

    csv_path = (
        out_root
        / "reports"
        / model_dir_name
        / "robustness"
        / "local"
        / "logitlens_local"
        / task
        / "tables"
        / f"{task}_local_logitlens_robustness_per_layer.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"Robustness-per-layer CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"layer", "robustness_auc"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} is missing required columns: {required}")

    df = df.sort_values("layer").reset_index(drop=True)
    model_name = pretty_model_name(model_dir_name)

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    ax.plot(
        df["layer"],
        df["robustness_auc"],
        marker="o",
        markersize=5,
        linewidth=2.5,
    )

    peak_idx = df["robustness_auc"].idxmax()
    peak_layer = int(df.loc[peak_idx, "layer"])
    peak_val = float(df.loc[peak_idx, "robustness_auc"])

    ax.scatter([peak_layer], [peak_val], s=60, zorder=3)
    ax.annotate(
        f"Peak: L{peak_layer}",
        xy=(peak_layer, peak_val),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=11,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Robustness AUC")
    ax.set_title(f"{task.upper()} local logit-lens robustness by layer\n{model_name}")
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.02, y=0.08)

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / f"{task}_local_logitlens_robustness_per_layer_{model_dir_name}.png"
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Example: out")
    ap.add_argument("--tasks", nargs="+", default=["mcq", "single"])
    ap.add_argument(
        "--save_dir",
        default=None,
        help="Default: <out_root>/reports/figures/logitlens_robustness_cross_model_figures",
    )
    ap.add_argument(
        "--noise_curve_models",
        nargs="*",
        default=[],
        help="Optional specific model dir names like Qwen__Qwen2.5-0.5B OPT-125M",
    )
    ap.add_argument(
        "--noise_curve_layers",
        nargs="*",
        type=int,
        default=None,
        help="Optional layers for per-model ACC-vs-sigma plots",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    save_dir = (
        Path(args.save_dir)
        if args.save_dir
        else (out_root / "reports" / "figures" / "logitlens_robustness_cross_model_figures")
    )

    for task in args.tasks:
        plot_robustness_vs_depth(out_root, task, save_dir)

        for model_dir_name in args.noise_curve_models:
            model_save_dir = (
                Path(args.save_dir)
                if args.save_dir
                else (
                    out_root
                    / "reports"
                    / model_dir_name
                    / "robustness"
                    / "local"
                    / "logitlens_local"
                    / task
                    / "figs"
                )
            )

            plot_noise_curves_for_one_model(
                out_root=out_root,
                task=task,
                model_dir_name=model_dir_name,
                save_dir=model_save_dir,
                layers=args.noise_curve_layers,
            )

            plot_robustness_per_layer_for_one_model(
                out_root=out_root,
                task=task,
                model_dir_name=model_dir_name,
                save_dir=model_save_dir,
            )


if __name__ == "__main__":
    main()