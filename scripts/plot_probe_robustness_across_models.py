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
        "Ammar-alhaj-ali/DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-R1-Distill (1.5B)",
    }
    return replacements.get(raw, raw)


def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 15,
    })


def find_robustness_csvs(out_root: Path, task: str, probe: str):
    pattern = (
        f"reports/*/robustness/local/probe_local/{task}/{probe}/tables/"
        f"probe_robustness_per_layer.csv"
    )
    return sorted(out_root.glob(pattern))


def _load_cross_model_probe_data(out_root: Path, task: str, probe: str):
    csvs = find_robustness_csvs(out_root, task, probe)
    if not csvs:
        return []

    curves = []
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

        curves.append((model_name, df))

    return curves


def plot_cross_model_robustness_vs_depth(out_root: Path, task: str, probe: str, save_dir: Path):
    apply_plot_style()

    curves = _load_cross_model_probe_data(out_root, task, probe)
    if not curves:
        raise FileNotFoundError(f"No probe robustness_per_layer CSVs found for task={task}, probe={probe}")

    fig, ax = plt.subplots(figsize=(8.6, 5.3))

    for model_name, df in curves:
        ax.plot(
            df["norm_depth"],
            df["robustness_auc"],
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=model_name,
        )

    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Robustness AUC")
    ax.set_title(f"Probe-space robustness vs depth ({task.upper()}, {probe})")
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    ax.legend(loc="best", frameon=True)
    ax.margins(x=0.02, y=0.08)

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / f"{task}_{probe}_probe_robustness_vs_depth_all_models.png"
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_png}")


def plot_probe_pair_comparison(out_root: Path, task: str, probes: list[str], save_dir: Path):
    """
    Create one combined figure with one subplot per probe.
    Intended for side-by-side figures like:
      left = linsvm
      right = logreg
    """
    apply_plot_style()

    if len(probes) < 2:
        print(f"[WARN] Need at least 2 probes for paired comparison, got {probes}")
        return

    selected_probes = probes[:2]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)

    plotted_any = False

    for ax, probe in zip(axes, selected_probes):
        curves = _load_cross_model_probe_data(out_root, task, probe)

        if not curves:
            ax.set_visible(False)
            continue

        plotted_any = True
        for model_name, df in curves:
            ax.plot(
                df["norm_depth"],
                df["robustness_auc"],
                linewidth=2.0,
                label=model_name,
            )

        ax.set_title(f"Probe-space robustness vs depth ({probe})")
        ax.set_xlabel("Normalized depth")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", labelrotation=0)
        ax.tick_params(axis="y", labelrotation=0)
        ax.margins(x=0.02, y=0.08)

        # legend inside each subplot, upper left usually works better here
        ax.legend(loc="best", frameon=True, fontsize=8)

    axes[0].set_ylabel("Robustness AUC")

    fig.suptitle(f"Probe-space robustness vs normalized depth in {task.upper()} task", y=1.02)
    plt.tight_layout()

    if plotted_any:
        save_dir.mkdir(parents=True, exist_ok=True)
        joined_probe_name = "_vs_".join(selected_probes)
        out_png = save_dir / f"{task}_{joined_probe_name}_probe_robustness_combined.png"
        plt.savefig(out_png, dpi=400, bbox_inches="tight")
        print(f"[SAVE] {out_png}")

    plt.close()


def plot_probe_noise_curves_for_one_model(
    out_root: Path,
    task: str,
    probe: str,
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
        / "probe_local"
        / task
        / probe
        / "tables"
        / "probe_acc_curves.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"Probe acc_curves CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"layer", "sigma", "acc"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} is missing required columns: {required}")

    df = df.sort_values(["layer", "sigma"]).reset_index(drop=True)

    if not layers:
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

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
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
    ax.set_title(f"Probe-space noise curves ({task.upper()}, {probe})")
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(chosen), 3),
        frameon=True,
        fontsize=10,
    )
    ax.margins(x=0.02, y=0.08)

    # model name داخل خود figure پایین راست
    ax.text(
        0.99, 0.02, model_name,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=10
    )

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / f"{task}_{probe}_probe_noise_curves_{model_dir_name}.png"
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_png}")


def plot_probe_robustness_per_layer_for_one_model(
    out_root: Path,
    task: str,
    probe: str,
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
        / "probe_local"
        / task
        / probe
        / "tables"
        / "probe_robustness_per_layer.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"Probe robustness_per_layer CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"layer", "robustness_auc"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} is missing required columns: {required}")

    df = df.sort_values("layer").reset_index(drop=True)
    model_name = pretty_model_name(model_dir_name)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(
        df["layer"],
        df["robustness_auc"],
        marker="o",
        markersize=4.5,
        linewidth=2.2,
    )

    peak_idx = df["robustness_auc"].idxmax()
    peak_layer = int(df.loc[peak_idx, "layer"])
    peak_val = float(df.loc[peak_idx, "robustness_auc"])
    ax.scatter([peak_layer], [peak_val], s=55, zorder=3)
    ax.annotate(
        f"Peak: L{peak_layer}",
        xy=(peak_layer, peak_val),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Robustness AUC")
    ax.set_title(f"{task.upper()} probe robustness by layer ({probe})")
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    ax.margins(x=0.02, y=0.08)

    ax.text(
        0.99, 0.02, model_name,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=10
    )

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / f"{task}_{probe}_probe_robustness_per_layer_{model_dir_name}.png"
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--tasks", nargs="+", default=["mcq", "single"])
    ap.add_argument("--probes", nargs="+", default=["linsvm", "logreg"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--noise_curve_layers", nargs="*", type=int, default=None)

    # new flags
    ap.add_argument(
        "--skip_single_plots",
        action="store_true",
        help="If set, do not generate the original single plots.",
    )
    ap.add_argument(
        "--make_combined_probe_plots",
        action="store_true",
        help="If set, also generate combined two-panel cross-model plots (e.g. linsvm vs logreg).",
    )

    args = ap.parse_args()

    out_root = Path(args.out_root)
    cross_save_dir = out_root / "reports" / "figures" / "probe_robustness_cross_model_figures"

    for task in args.tasks:
        if args.make_combined_probe_plots:
            plot_probe_pair_comparison(
                out_root=out_root,
                task=task,
                probes=args.probes,
                save_dir=cross_save_dir,
            )

        for probe in args.probes:
            if not args.skip_single_plots:
                plot_cross_model_robustness_vs_depth(out_root, task, probe, cross_save_dir)

                for model_dir_name in args.models:
                    model_save_dir = (
                        out_root
                        / "reports"
                        / model_dir_name
                        / "robustness"
                        / "local"
                        / "probe_local"
                        / task
                        / probe
                        / "figs"
                    )

                    plot_probe_noise_curves_for_one_model(
                        out_root=out_root,
                        task=task,
                        probe=probe,
                        model_dir_name=model_dir_name,
                        save_dir=model_save_dir,
                        layers=args.noise_curve_layers,
                    )

                    plot_probe_robustness_per_layer_for_one_model(
                        out_root=out_root,
                        task=task,
                        probe=probe,
                        model_dir_name=model_dir_name,
                        save_dir=model_save_dir,
                    )


if __name__ == "__main__":
    main()