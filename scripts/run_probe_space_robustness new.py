# scripts/run_probe_space_robustness.py
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.feature_cache import load_feature_cache
from src.linear_probes import fit_eval_probes_per_layer, tune_probes_on_layer


def normalized_auc(sigmas, accs):
    s = np.asarray(sigmas, float)
    a = np.asarray(accs, float)
    if len(s) < 2:
        return np.nan
    idx = np.argsort(s)
    s, a = s[idx], a[idx]
    area = np.trapezoid(a, s)
    denom = (s[-1] - s[0]) * 1.0
    return float(area / denom) if denom > 0 else np.nan


def question_level_winners(scores, qids):
    """
    Return the winning candidate index (within each qid group) under argmax(score).
    The returned dict maps qid -> global row index of the winning candidate.
    """
    scores = np.asarray(scores).reshape(-1)
    qids = np.asarray(qids)

    assert len(scores) == len(qids)

    order = np.argsort(qids, kind="stable")
    scores_ord = scores[order]
    qids_ord = qids[order]
    row_ord = order

    uniq, starts = np.unique(qids_ord, return_index=True)
    starts = list(starts) + [len(qids_ord)]

    winners = {}
    for i in range(len(uniq)):
        a, b = starts[i], starts[i + 1]
        local_argmax = int(np.argmax(scores_ord[a:b]))
        winners[uniq[i]] = int(row_ord[a + local_argmax])
    return winners


def question_level_acc(scores, y, qids):
    """
    Group by qid and select the candidate with the highest score in each group.
    Accuracy is the fraction of questions for which the winner has y == 1.
    """
    scores = np.asarray(scores).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)
    qids = np.asarray(qids)

    assert len(scores) == len(y) == len(qids)

    winners = question_level_winners(scores, qids)
    if not winners:
        return np.nan

    correct = 0
    total = 0
    for _, row_idx in winners.items():
        correct += int(y[row_idx] == 1)
        total += 1
    return float(correct / total) if total > 0 else np.nan


def question_level_flip_rate(base_scores, noisy_scores, qids):
    """
    Fraction of qid groups for which the winning candidate changes after noise.
    """
    base_w = question_level_winners(base_scores, qids)
    noisy_w = question_level_winners(noisy_scores, qids)

    keys = sorted(set(base_w.keys()) & set(noisy_w.keys()))
    if not keys:
        return np.nan

    flips = sum(int(base_w[k] != noisy_w[k]) for k in keys)
    return float(flips / len(keys))


def main():
    ap = argparse.ArgumentParser("Probe-space robustness vs σ per layer (question-level)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["mcq", "single"], required=True)
    ap.add_argument("--out_root", default=str(REPO_ROOT / "out"))
    ap.add_argument(
        "--sigmas",
        default="0.0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2"
    )
    ap.add_argument("--noise_mode", choices=["rel", "abs"], default="rel")
    ap.add_argument("--method", choices=["logreg", "linsvm"], default="logreg")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    model_path = args.model.replace("/", "__")
    cache_dir = Path(args.out_root) / "features" / model_path / args.task

    Xtr_layers, ytr, qtr = load_feature_cache(cache_dir / "train.npz")
    Xte_layers, yte, qte = load_feature_cache(cache_dir / "test.npz")

    layers = sorted(Xtr_layers.keys())
    sigmas = sorted(set(float(x) for x in args.sigmas.split(",") if x.strip()))
    if 0.0 not in sigmas:
        sigmas = [0.0] + sigmas

    # Tune on one middle layer, then reuse across all layers
    tuned = None
    mid = layers[len(layers) // 2]
    tuned = tune_probes_on_layer(
        Xtr_layers[mid].astype(np.float32), ytr, method=args.method
    )

    rows = []
    base_scores_by_layer = {}

    # Baseline on clean test
    base_res = fit_eval_probes_per_layer(
        {li: Xtr_layers[li].astype(np.float32) for li in layers}, ytr,
        {li: Xte_layers[li].astype(np.float32) for li in layers}, yte,
        method=args.method, best_params=tuned
    )

    for li in layers:
        r0 = base_res.get(li, {})
        s0 = np.asarray(r0.get("scores", []), dtype=float).reshape(-1)
        base_scores_by_layer[li] = s0
        acc_q = question_level_acc(s0, yte, qte) if s0.size else np.nan
        rows.append({
            "layer": li,
            "sigma": 0.0,
            "acc": acc_q,
            "flip_rate": 0.0
        })

    # Sweep sigmas
    rng = np.random.RandomState(args.seed)

    for li in layers:
        Xtr = Xtr_layers[li].astype(np.float32, copy=False)
        Xte = Xte_layers[li].astype(np.float32, copy=False)

        # Relative or absolute noise scale
        if args.noise_mode == "rel":
            S = Xtr.std(axis=0, keepdims=True)
            S[~np.isfinite(S)] = 1.0
            S[S == 0] = 1.0
        else:
            S = 1.0

        # Clipping bounds from clean training statistics
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True)
        sd[~np.isfinite(sd)] = 1.0
        sd[sd == 0] = 1.0
        lo = mu - 6.0 * sd
        hi = mu + 6.0 * sd

        for sig in sigmas:
            if sig == 0.0:
                continue

            noise = rng.normal(0.0, sig, size=Xte.shape).astype(np.float32) * S
            Xn = Xte + noise
            Xn = np.minimum(np.maximum(Xn, lo), hi)
            Xn = np.nan_to_num(Xn, nan=0.0, posinf=hi, neginf=lo)

            res = fit_eval_probes_per_layer(
                {li: Xtr}, ytr,
                {li: Xn}, yte,
                method=args.method, best_params=tuned
            )

            if li in res and "scores" in res[li]:
                s_noisy = np.asarray(res[li]["scores"], dtype=float).reshape(-1)
                acc_q = question_level_acc(s_noisy, yte, qte)
                flip_q = question_level_flip_rate(base_scores_by_layer[li], s_noisy, qte)
            else:
                acc_q = np.nan
                flip_q = np.nan

            rows.append({
                "layer": li,
                "sigma": sig,
                "acc": acc_q,
                "flip_rate": flip_q
            })

    # Build dataframe
    df = pd.DataFrame(rows).sort_values(["layer", "sigma"])

    sub = "probe_local"
    out_dir = Path(args.out_root) / "reports" / model_path / "robustness" / "local" / sub / args.task / args.method
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "tables" / "probe_acc_curves.csv", index=False)

    # Per-layer AUC
    auc_rows = []
    for li in layers:
        sub_df = df[df["layer"] == li]
        auc = normalized_auc(sub_df["sigma"].tolist(), sub_df["acc"].tolist())
        flip_auc = normalized_auc(sub_df["sigma"].tolist(), sub_df["flip_rate"].tolist())
        auc_rows.append({
            "layer": li,
            "robustness_auc": auc,
            "flip_auc": flip_auc
        })

        # Accuracy curve
        plt.figure(figsize=(5, 3.2))
        plt.plot(sub_df["sigma"], sub_df["acc"], marker="o")
        plt.xlabel("σ")
        plt.ylabel("Question-level accuracy")
        plt.title(f"Layer {li}: probe-space robustness ({args.method})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "figs" / f"acc_vs_sigma_layer{li}.png", dpi=180)
        plt.close()

        # Flip-rate curve
        plt.figure(figsize=(5, 3.2))
        plt.plot(sub_df["sigma"], sub_df["flip_rate"], marker="o")
        plt.xlabel("σ")
        plt.ylabel("Question-level flip rate")
        plt.title(f"Layer {li}: probe-space flip rate ({args.method})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "figs" / f"flip_vs_sigma_layer{li}.png", dpi=180)
        plt.close()

    pd.DataFrame(auc_rows).to_csv(out_dir / "tables" / "probe_robustness_per_layer.csv", index=False)

    plt.figure(figsize=(6, 3.2))
    plt.plot([r["layer"] for r in auc_rows], [r["robustness_auc"] for r in auc_rows], marker="s")
    plt.xlabel("Layer")
    plt.ylabel("Question-level robustness (normalised AUC)")
    plt.title(f"Probe-space robustness vs layer ({args.method})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "figs" / "robustness_vs_layer.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 3.2))
    plt.plot([r["layer"] for r in auc_rows], [r["flip_auc"] for r in auc_rows], marker="s")
    plt.xlabel("Layer")
    plt.ylabel("Question-level flip-rate AUC")
    plt.title(f"Probe-space flip-rate summary vs layer ({args.method})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "figs" / "flip_vs_layer.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()