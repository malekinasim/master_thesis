#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def pick_col(cols, candidates):
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    return None

def main():
    ap = argparse.ArgumentParser("Compute robustness per layer as AUC of ACC vs sigma")
    ap.add_argument("--in_csv", required=True, help="CSV with columns: layer, sigma, acc (optionally rep)")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--title", default=None)
    ap.add_argument("--normalize", action="store_true",
                    help="Normalize AUC by sigma range: (1/(sigma_max-sigma_min))*∫Acc dσ")
    ap.add_argument("--relative", action="store_true",
                    help="Compute relative robustness by dividing Acc(σ) by Acc(0) per layer before integrating")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    # auto-detect columns
    layer_c = pick_col(df.columns, ["layer", "layer_idx", "l"])
    sigma_c = pick_col(df.columns, ["sigma", "noise", "noise_std", "std"])
    acc_c   = pick_col(df.columns, ["acc", "accuracy"])

    if layer_c is None or sigma_c is None or acc_c is None:
        raise ValueError(f"Could not find required columns. Have: {list(df.columns)}")

    # average repetitions if present
    rep_c = pick_col(df.columns, ["rep", "repeat", "seed"])
    if rep_c is not None:
        df = df.groupby([layer_c, sigma_c], as_index=False)[acc_c].mean()

    df[layer_c] = df[layer_c].astype(int)
    df = df.sort_values([layer_c, sigma_c])

    rows = []
    for layer, g in df.groupby(layer_c):
        g = g.sort_values(sigma_c)
        x = g[sigma_c].to_numpy(dtype=float)
        y = g[acc_c].to_numpy(dtype=float)

        # baseline at sigma=0 (or min sigma if 0 not present)
        s0 = float(np.min(x))
        acc0 = float(np.mean(y[x == s0])) if np.any(x == s0) else float(y[0])

        if args.relative:
            y = y / (acc0 + 1e-12)

        auc = float(np.trapz(y, x))
        if args.normalize:
            rng = float(np.max(x) - np.min(x))
            auc = auc / rng if rng > 0 else np.nan

        rows.append({
            "layer": int(layer),
            "sigma_min": float(np.min(x)),
            "sigma_max": float(np.max(x)),
            "acc_sigma0": acc0,
            "robustness_auc": auc,
        })

    out = pd.DataFrame(rows).sort_values("layer")
    out_csv = out_dir / "robustness_by_layer.csv"
    out.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}")

    # plot robustness vs layer
    plt.figure(figsize=(10, 6))
    plt.plot(out["layer"], out["robustness_auc"], marker="s")
    plt.xlabel("Layer")
    plt.ylabel("Robustness (AUC of ACC vs sigma)" + (" (normalized)" if args.normalize else ""))
    plt.title(args.title or "Robustness per layer")
    plt.grid(alpha=0.25)
    out_png = out_dir / "robustness_vs_layer.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[SAVE] {out_png}")

if __name__ == "__main__":
    main()
