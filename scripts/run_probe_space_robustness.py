# scripts/run_probe_space_robustness.py
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.feature_cache import load_feature_cache                    # :contentReference[oaicite:17]{index=17}
from src.linear_probes import fit_eval_probes_per_layer, tune_probes_on_layer  # :contentReference[oaicite:18]{index=18}
from src.metrics import roc_auc                                      # :contentReference[oaicite:19]{index=19}

def normalized_auc(sigmas, accs):
    s = np.asarray(sigmas, float); a = np.asarray(accs, float)
    if len(s) < 2: return np.nan
    idx = np.argsort(s); s, a = s[idx], a[idx]
    area = np.trapezoid(a, s)
    denom = (s[-1] - s[0]) * 1.0
    return float(area / denom) if denom > 0 else np.nan

def main():
    ap = argparse.ArgumentParser("Probe‑space (T2) robustness vs σ per layer")
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["mcq","single"], required=True)
    ap.add_argument("--out_root", default=str(REPO_ROOT / "out"))
    ap.add_argument("--sigmas", default="0.0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2")
    ap.add_argument("--noise_mode", choices=["rel","abs"], default="rel")
    ap.add_argument("--method", choices=["massmean","lda","logreg","linsvm"], default="lda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    model_path = args.model.replace("/", "__")
    cache_dir = Path(args.out_root) / "features" / model_path / args.task
    Xtr_layers, ytr, qtr = load_feature_cache(cache_dir / "train.npz")   # :contentReference[oaicite:20]{index=20}
    Xte_layers, yte, qte = load_feature_cache(cache_dir / "test.npz")

    layers = sorted(Xtr_layers.keys())
    sigmas = sorted(set(float(x) for x in args.sigmas.split(",") if x.strip()))
    if 0.0 not in sigmas: sigmas = [0.0] + sigmas

    # tune probe on یک لایهٔ میانی (برای روش‌های یادگرفتنی)
    tuned = None
    mid = layers[len(layers)//2]
    if args.method in {"lda","logreg","linsvm"}:
        tuned = tune_probes_on_layer(Xtr_layers[mid].astype(np.float32), ytr, method=args.method)  # :contentReference[oaicite:21]{index=21}

    rows = []
    # baseline on clean test
    base_res = fit_eval_probes_per_layer(
        {li: Xtr_layers[li].astype(np.float32) for li in layers}, ytr,
        {li: Xte_layers[li].astype(np.float32) for li in layers}, yte,
        method=args.method, best_params=tuned
    )  # خروجی شامل 'scores' است تا Acc@Q و AUROC را بشود محاسبه کرد. :contentReference[oaicite:22]{index=22}

    for li in layers:
        r0 = base_res.get(li, {})
        rows.append({"layer": li, "sigma": 0.0, "acc": float(r0.get("acc", np.nan))})

    # sweep sigmas
    rng = np.random.RandomState(args.seed)
    for li in layers:
        Xtr = Xtr_layers[li].astype(np.float32, copy=False)
        Xte = Xte_layers[li].astype(np.float32, copy=False)

        # scale for rel/abs
        S = Xtr.std(axis=0, keepdims=True) if args.noise_mode=="rel" else 1.0
        S[~np.isfinite(S)] = 1.0; S[S==0] = 1.0

        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True); sd[~np.isfinite(sd)] = 1.0; sd[sd==0]=1.0
        lo = mu - 6.0*sd; hi = mu + 6.0*sd

        for sig in sigmas:
            if sig == 0.0: 
                continue
            noise = rng.normal(0.0, sig, size=Xte.shape).astype(np.float32) * S
            Xn = Xte + noise
            Xn = np.minimum(np.maximum(Xn, lo), hi)
            Xn = np.nan_to_num(Xn, nan=0.0, posinf=hi, neginf=lo)

            res = fit_eval_probes_per_layer({li: Xtr}, ytr, {li: Xn}, yte, method=args.method, best_params=tuned)
            acc = float(res[li]["acc"]) if li in res else np.nan
            rows.append({"layer": li, "sigma": sig, "acc": acc})

    # build DF
    df = pd.DataFrame(rows).sort_values(["layer","sigma"])

    sub = "probe_local"
    out_dir = Path(args.out_root)/ "reports"/ model_path/"robustness"/"local"/sub /args.task/args.method

    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tables" / "probe_acc_curves.csv", index=False)

    # per-layer AUC
    auc_rows = []
    for li in layers:
        sub = df[df["layer"]==li]
        auc = normalized_auc(sub["sigma"].tolist(), sub["acc"].tolist())
        auc_rows.append({"layer": li, "robustness_auc": auc})
        # plot curve
        plt.figure(figsize=(5,3.2))
        plt.plot(sub["sigma"], sub["acc"], marker="o")
        plt.xlabel("σ"); plt.ylabel("Accuracy")
        plt.title(f"Layer {li}: probe-space ({args.method})")
        plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(out_dir / "figs" / f"acc_vs_sigma_layer{li}.png", dpi=180); plt.close()

    pd.DataFrame(auc_rows).to_csv(out_dir / "tables" / "probe_robustness_per_layer.csv", index=False)
    plt.figure(figsize=(6,3.2))
    plt.plot([r["layer"] for r in auc_rows], [r["robustness_auc"] for r in auc_rows], marker="s")
    plt.xlabel("Layer"); plt.ylabel("Robustness (AUCσ normalized)")
    plt.title(f"Probe-space ({args.method}) robustness vs layer")
    plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out_dir / "figs" / "robustness_vs_layer.png", dpi=180); plt.close()

if __name__ == "__main__":
    main()
