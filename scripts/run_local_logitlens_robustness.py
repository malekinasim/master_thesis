import os, sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.io import load_prompts_with_options
from src.util import load_model_and_tokenizer
from src.hooks import _alllayer_tokenpos_hiddens_core


def try_get_blocks(model: nn.Module):
    candidates = [
        ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
        ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("gpt_neox.layers", lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
        ("model.decoder.layers", lambda m: getattr(getattr(getattr(m, "model", None), "decoder", None), "layers", None)),
    ]
    for name, getter in candidates:
        blocks = getter(model)
        if blocks is not None:
            if isinstance(blocks, (nn.ModuleList, list)):
                blocks = list(blocks)
            else:
                try:
                    blocks = list(blocks)
                except TypeError:
                    continue
            if len(blocks) > 0:
                print(f"[info] using blocks from {name}: n_layers={len(blocks)}")
                return blocks
    raise RuntimeError("Cannot locate transformer blocks on this model")


def _robust_std_vector(X, q_clip=(5, 95), ddof=0):
    s = np.asarray(X, dtype=np.float32).std(axis=0, ddof=ddof)
    s[~np.isfinite(s)] = np.nan
    pos = s[(s > 0) & np.isfinite(s)]
    if pos.size == 0:
        s_typ = float(np.nanstd(X))
        if not np.isfinite(s_typ) or s_typ == 0.0:
            s_typ = 1.0
        return torch.from_numpy(np.full_like(s, s_typ, dtype=np.float32))
    med = float(np.nanmedian(pos))
    s = np.where(np.isfinite(s) & (s > 0), s, med)
    if q_clip is not None:
        lo, hi = np.nanpercentile(pos, q_clip)
        s = np.clip(s, lo, hi)
    return torch.from_numpy(s.astype(np.float32))


def _merge_prompt_completion(prompt: str, completion: str, add_space_between: bool = True):
    p = str(prompt)
    c = str(completion)
    need_bridge = (add_space_between and c and not c[:1].isspace() and not p.endswith((" ", "\t", "\n")))
    if need_bridge:
        return p + " " + c, len(p) + 1
    else:
        return p + c, len(p)


def first_token_id_in_context(tok, prompt: str, completion: str, add_space_between: bool = True) -> int:
    txt, start = _merge_prompt_completion(prompt, completion, add_space_between=add_space_between)
    enc = tok(
        txt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    ids = enc["input_ids"][0].tolist()
    offs = enc["offset_mapping"][0].tolist()
    for i, (a, b) in enumerate(offs):
        if (b > start) and (a >= start):
            return int(ids[i])
    return int(ids[-1])


def option_first_token_ids_in_context(tok, prompt: str, options: list[str]) -> list[int]:
    return [first_token_id_in_context(tok, prompt, o) for o in options]


@torch.no_grad()
def resolve_ln_and_WU(model: nn.Module):
    ln_candidates = [
        getattr(getattr(model, "transformer", None), "ln_f", None),
        getattr(model, "ln_f", None),
        getattr(getattr(model, "model", None), "norm", None),
        getattr(getattr(model, "model", None), "final_layernorm", None),
        getattr(model, "norm", None),
    ]
    ln = None
    for cur in ln_candidates:
        if cur is not None:
            ln = cur
            break
    lm_head = getattr(model, "lm_head", None)
    assert (lm_head is not None) and hasattr(lm_head, "weight"), "model.lm_head.weight missing"
    WU = lm_head.weight.T.contiguous()
    return ln, WU


@torch.no_grad()
def _project_logits_from_h(h, ln, WU):
    dev = WU.device
    x = h.to(dev)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if ln is not None:
        ln = ln.to(dev)
        x = ln(x.unsqueeze(1))[:, -1, :]
    z = x @ WU
    return z if h.ndim == 2 else z[0]


def normalized_auc(sigmas, accs):
    x = np.asarray(sigmas, dtype=float)
    y = np.asarray(accs, dtype=float)
    if len(x) < 2:
        return float("nan")
    order = np.argsort(x)
    x, y = x[order], y[order]
    area = np.trapz(y, x)
    denom = (x[-1] - x[0])
    return float(area / denom) if denom > 0 else float("nan")


@torch.no_grad()
def batched_promptend_hiddens(model, tok, prompt_texts, batch_size=64, skip_embedding=True):
    all_layers = None
    for i in range(0, len(prompt_texts), batch_size):
        chunk = prompt_texts[i:i+batch_size]
        layer_chunk = _alllayer_tokenpos_hiddens_core(
            model,
            tok,
            chunk,
            pos_spec=-1,
            skip_embedding=skip_embedding,
            comp_starts_char=None,
            need_offsets=False,
        )
        if all_layers is None:
            all_layers = [lc for lc in layer_chunk]
        else:
            for li in range(len(all_layers)):
                all_layers[li] = torch.cat([all_layers[li], layer_chunk[li]], dim=0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_layers


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Local raw Logit Lens robustness from prompt-only states")
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["mcq", "single"], required=True)
    ap.add_argument("--dataset_root", default=str(REPO_ROOT / "data"))
    ap.add_argument("--out_root", default=str(REPO_ROOT / "out"))
    ap.add_argument("--sigmas", default="0.0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2")
    ap.add_argument("--noise_mode", choices=["rel", "abs"], default="rel")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, tok = load_model_and_tokenizer(args.model, device=device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_path = args.model.replace("/", "__")
    ds_path = Path(args.dataset_root) / model_path / "prompt_pool.json"
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")

    mcq_items, single_items = load_prompts_with_options(str(ds_path), tok, require_single_token=False)

    if args.task == "mcq":
        items = [
            it for it in mcq_items
            if isinstance(it.get("question"), str)
            and isinstance(it.get("options"), list)
            and it.get("answer") in it["options"]
        ]
    else:
        items = [
            it for it in single_items
            if isinstance(it.get("question"), str)
            and isinstance(it.get("answer"), str)
        ]

    if args.max_items is not None:
        items = items[:args.max_items]
    if not items:
        raise RuntimeError("No valid items after filtering.")

    print(f"[info] #items={len(items)}")

    blocks = try_get_blocks(model)
    L = len(blocks)

    prompts = []
    opt_ids_per_question = []
    gold_idx = []
    gold_tok_ids = []

    if args.task == "mcq":
        for it in items:
            q = it["question"]
            options = it["options"]
            gold = it["answer"]

            prompts.append(q)
            opt_ids_per_question.append(option_first_token_ids_in_context(tok, q, options))
            gold_idx.append(options.index(gold))
    else:
        for it in items:
            q = it["question"]
            gold = it["answer"]

            prompts.append(q)
            gold_tok_ids.append(first_token_id_in_context(tok, q, gold))

    Nq = len(prompts)
    print(f"[info] building prompt-only hidden states for Nq={Nq} questions")

    layer_h = batched_promptend_hiddens(
        model,
        tok,
        prompts,
        batch_size=max(1, args.batch_size),
        skip_embedding=True,
    )

    ln_f, WU = resolve_ln_and_WU(model)

    baseline_pred = {l: [] for l in range(L)}
    for li in range(L):
        H = layer_h[li].to(device)
        z = _project_logits_from_h(H, ln_f, WU)

        if args.task == "single":
            baseline_pred[li] = torch.argmax(z, dim=-1).tolist()
        else:
            preds = []
            for i_q, opt_ids in enumerate(opt_ids_per_question):
                slice_logits = [float(z[i_q, tid].item()) for tid in opt_ids]
                preds.append(int(np.argmax(slice_logits)))
            baseline_pred[li] = preds

    sigmas = sorted(set(float(x) for x in args.sigmas.split(",") if x.strip()))
    if 0.0 not in sigmas:
        sigmas = [0.0] + sigmas

    sub = "logitlens_local"
    base_dir = Path(args.out_root) / "reports" / model_path / "robustness" / "local" / sub / args.task
    (base_dir / "figs").mkdir(parents=True, exist_ok=True)
    (base_dir / "tables").mkdir(parents=True, exist_ok=True)

    layer_auc = []
    rows = []

    for li in range(L):
        H = layer_h[li].to(device)
        D = H.shape[1]

        if args.noise_mode == "rel":
            per_dim_std = _robust_std_vector(layer_h[li].numpy()).to(device=device, dtype=torch.float32)
        else:
            per_dim_std = torch.ones(D, dtype=torch.float32, device=device)

        acc_curve, sig_curve = [], []

        for sig in sigmas:
            base_seed = args.seed + 1000 * li + int(sig * 1e3)

            if sig == 0.0:
                pred = baseline_pred[li]
            else:
                if args.task == "mcq":
                    wins = []
                else:
                    preds_repeats = []

                for r in range(max(1, args.repeats)):
                    g_dev = torch.Generator(device=device).manual_seed(base_seed + r)
                    eps = torch.randn((Nq, D), device=device, generator=g_dev) * (sig * per_dim_std)
                    Hn = H + eps
                    z = _project_logits_from_h(Hn, ln_f, WU)

                    if args.task == "single":
                        preds_repeats.append(torch.argmax(z, dim=-1).tolist())
                    else:
                        preds_r = []
                        for i_q, opt_ids in enumerate(opt_ids_per_question):
                            slice_logits = [float(z[i_q, tid].item()) for tid in opt_ids]
                            preds_r.append(int(np.argmax(slice_logits)))
                        wins.append(preds_r)

                if args.task == "single":
                    if len(preds_repeats) > 1:
                        arr = np.array(preds_repeats)
                        voted = []
                        for q in range(arr.shape[1]):
                            vals, counts = np.unique(arr[:, q], return_counts=True)
                            voted.append(int(vals[np.argmax(counts)]))
                        pred = voted
                    else:
                        pred = preds_repeats[0]
                else:
                    if len(wins) > 1:
                        arr = np.array(wins)
                        voted = []
                        for q in range(arr.shape[1]):
                            vals, counts = np.unique(arr[:, q], return_counts=True)
                            voted.append(int(vals[np.argmax(counts)]))
                        pred = voted
                    else:
                        pred = wins[0]

            if args.task == "mcq":
                correct_cnt = 0
                flip_cnt = 0
                for i_q, g_idx in enumerate(gold_idx):
                    if pred[i_q] == g_idx:
                        correct_cnt += 1
                    if pred[i_q] != baseline_pred[li][i_q]:
                        flip_cnt += 1
                acc = correct_cnt / max(Nq, 1)
                flip = flip_cnt / max(Nq, 1)
            else:
                correct_cnt = 0
                flip_cnt = 0
                for i_q in range(Nq):
                    if pred[i_q] == gold_tok_ids[i_q]:
                        correct_cnt += 1
                    if pred[i_q] != baseline_pred[li][i_q]:
                        flip_cnt += 1
                acc = correct_cnt / max(Nq, 1)
                flip = flip_cnt / max(Nq, 1)

            acc_curve.append(acc)
            sig_curve.append(sig)
            rows.append({
                "layer": li,
                "sigma": sig,
                "acc": acc,
                "flip_rate": flip,
            })

        auc_val = normalized_auc(sig_curve, acc_curve)
        layer_auc.append({"layer": li, "robustness_auc": auc_val})

        plt.figure(figsize=(5, 4))
        order = np.argsort(sig_curve)
        xs = np.array(sig_curve)[order]
        ys = np.array(acc_curve)[order]
        plt.plot(xs, ys, marker="o")
        plt.xlabel("σ (noise std)")
        plt.ylabel("Local decoding accuracy")
        plt.title(f"Layer {li}: ACC vs σ ({args.task})")
        plt.grid(alpha=0.3)
        out_png = base_dir / "figs" / f"{args.task}_acc_vs_sigma_layer{li}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[SAVE] {out_png}")

    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = base_dir / "tables" / f"{args.task}_local_logitlens_noise_curves.csv"
    df.to_csv(out_csv, index=False)
    print("[SAVE]", out_csv)

    df_auc = pd.DataFrame(layer_auc).sort_values("layer")
    out_csv2 = base_dir / "tables" / f"{args.task}_local_logitlens_robustness_per_layer.csv"
    df_auc.to_csv(out_csv2, index=False)
    print("[SAVE]", out_csv2)

if __name__ == "__main__":
    main()