# scripts/run_local_logitlens_robustness.py
import os, sys, argparse, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# --- project utils
from src.io import load_prompts_with_options
from src.util import load_model_and_tokenizer
from src.feature_cache import load_feature_cache
from src.hooks import _alllayer_lasttoken_hiddens_core

# =================== Helper: which blocks are "layers" ===================
def try_get_blocks(model: nn.Module):
    """
    Try to locate the list of transformer blocks (layers) on HF models.
    Supports GPT2/OPT/Neo/Llama-like layouts.
    """
    # common paths
    candidates = [
        ("transformer.h",      lambda m: getattr(getattr(m, "transformer", None), "h", None)),
        ("model.layers",       lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("gpt_neox.layers",    lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
        ("model.decoder.layers", lambda m: getattr(getattr(m, "model", None), "decoder", None)),
    ]
    for name, getter in candidates:
        blocks = getter(model)
        if blocks is not None:
            if isinstance(blocks, nn.ModuleList) or isinstance(blocks, list):
                blocks = list(blocks)
            else:
                try:
                    blocks = list(blocks)
                except TypeError:
                    continue
            if len(blocks) > 0:
                print(f"[info] using blocks from {name}: n_layers={len(blocks)}")
                return blocks

    # heuristic fallback: anything that looks like a residual block with attention/mlp
    blocks = []
    for m in model.modules():
        if hasattr(m, "self_attn") or hasattr(m, "attention") or hasattr(m, "mlp"):
            if isinstance(m, nn.Module) and len(list(m.children())) > 0:
                blocks.append(m)
    if blocks:
        print(f"[warn] using heuristic blocks, n={len(blocks)}")
        return blocks
    raise RuntimeError("Cannot locate transformer blocks on this model")

# =================== Helper: robust std vector for rel noise =============
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

def get_layer_std_from_cache(out_root, model_path, task):
    """
    Load per-layer per-dim std from training cache (used in 'rel' noise).
    Falls back to None if cache absent.
    """
    cache_path = Path(out_root) / "features" / model_path / task / "train.npz"
    if not cache_path.exists():
        print(f"[warn] training cache not found: {cache_path}")
        return None
    X_layers, _, _ = load_feature_cache(cache_path)
    stds = {}
    for li, X in X_layers.items():
        stds[int(li)] = _robust_std_vector(X)
    return stds

# =================== Helper: tokenization in context =====================
def _merge_prompt_completion(prompt: str, completion: str, add_space_between: bool = True):
    p = str(prompt)
    c = str(completion)
    need_bridge = (add_space_between and c and not c[:1].isspace() and not p.endswith((" ", "\t", "\n")))
    if need_bridge:
        return p + " " + c, len(p) + 1
    else:
        return p + c, len(p)

def first_token_id_in_context(tok, prompt: str, completion: str, add_space_between: bool = True) -> int:
    """
    Return *first* token id of completion in the context (prompt+completion).
    """
    txt, start = _merge_prompt_completion(prompt, completion, add_space_between=add_space_between)
    enc = tok(
        txt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    ids  = enc["input_ids"][0].tolist()
    offs = enc["offset_mapping"][0].tolist()
    for i, (a, b) in enumerate(offs):
        if (b > start) and (a >= start):
            return int(ids[i])
    # fallback: آخرین توکن
    return int(ids[-1])

def option_first_token_ids_in_context(tok, prompt: str, options: list[str]) -> list[int]:
    return [first_token_id_in_context(tok, prompt, o) for o in options]

# ---------------- Resolve final LayerNorm & W_U once -------------------
@torch.no_grad()
def resolve_ln_and_WU(model: nn.Module):
    """
    Try to get the final LayerNorm (if present) and the unembedding matrix W_U
    from a HF causal LM. Supports GPT2/OPT/Neo/Llama-like.
    """
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
    WU = lm_head.weight.T.contiguous()   # [d, V]
    return ln, WU

@torch.no_grad()
def _project_logits_from_h(h, ln, WU):
    """
    h: [B, D] or [D]
    returns logits: [B,V] or [V]

    مهم: همه‌ی تنسورها روی همان device وزن‌های WU قرار می‌گیرند
    تا خطای cuda/cpu نداشته باشیم.
    """
    dev = WU.device

    # h را روی همان device می‌بریم
    x = h.to(dev)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # [1, D]

    # لایه‌ی نرمال‌سازی نهایی (در صورت وجود)
    if ln is not None:
        ln = ln.to(dev)
        # ln معمولاً برای [B,T,D] تعریف شده؛ ما با T=1 فیک صدا می‌زنیم
        x = ln(x.unsqueeze(1))[:, -1, :]  # [B,D]

    z = x @ WU  # [B,V]
    return z if h.ndim == 2 else z[0]

# =================== Robustness metric: AUC over σ ======================
def normalized_auc(sigmas, accs):
    """
    Trapezoidal AUC normalized to [0,1] by dividing by (sigma_max - sigma_min).
    Requires sigma sorted ascending. Includes sigma=0 baseline.
    """
    x = np.asarray(sigmas, dtype=float)
    y = np.asarray(accs,  dtype=float)
    if len(x) < 2:
        return float("nan")
    order = np.argsort(x)
    x, y = x[order], y[order]
    area = np.trapz(y, x)
    denom = (x[-1] - x[0])
    return float(area / denom) if denom > 0 else float("nan")

# =================== Batched hidden extraction to avoid OOM =============
@torch.no_grad()
def batched_lasttoken_hiddens(model, tok, merged_texts, batch_size=64, skip_embedding=True):
    """
    Compute per-layer last-token hidden states in batches to reduce memory.
    Returns List[L] where each element is [B,D] on CPU.
    """
    all_layers = None
    for i in range(0, len(merged_texts), batch_size):
        chunk = merged_texts[i:i+batch_size]
        layer_chunk = _alllayer_lasttoken_hiddens_core(model, tok, chunk, skip_embedding=skip_embedding)
        if all_layers is None:
            all_layers = [lc for lc in layer_chunk]
        else:
            for li in range(len(all_layers)):
                all_layers[li] = torch.cat([all_layers[li], layer_chunk[li]], dim=0)
        torch.cuda.empty_cache()
    return all_layers

# =================== Main ==============================================
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Local logit-lens robustness vs hidden noise")
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--task", choices=["mcq", "single"], required=True)
    ap.add_argument("--dataset_root", default=str(REPO_ROOT / "data"))
    ap.add_argument("--out_root",     default=str(REPO_ROOT / "out"))
    ap.add_argument("--sigmas",       default="0.0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2")
    ap.add_argument("--noise_mode",   choices=["rel", "abs"], default="rel")
    ap.add_argument("--repeats",      type=int, default=1)
    ap.add_argument("--max_items",    type=int, default=None,
                    help="Optional cap on number of questions for speed")
    ap.add_argument("--batch_size",   type=int, default=64,
                    help="Batch size for hidden extraction to avoid OOM")
    ap.add_argument("--device",       default="auto")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--single_pos_only", action="store_true",
                    help="For SINGLE task, only add noise at the last token.")
    args = ap.parse_args()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # load model/tokenizer
    model, tok = load_model_and_tokenizer(args.model, device=device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_path = args.model.replace("/", "__")
    # dataset
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
            and isinstance(it.get("answer"),  str)
        ]

    if args.max_items is not None:
        items = items[: args.max_items]

    if not items:
        raise RuntimeError("No valid items after filtering.")

    print(f"[info] #items={len(items)}")

    # get blocks & n_layers
    blocks = try_get_blocks(model)
    L = len(blocks)

    # gather hidden states for last token of each (prompt+option) or prompt+answer
    # using hooks._alllayer_lasttoken_hiddens_core
    prompts = []
    comps   = []
    labels  = []
    comp_tok_ids = []  # first-token ids for each (prompt, completion) pair
    opt_ids_per_question = []

    if args.task == "mcq":
        for it in items:
            q, options, gold = it["question"], it["options"], it["answer"]
            # For each option, we treat it as a (prompt, completion) pair
            opt_ids = option_first_token_ids_in_context(tok, q, options)
            for j, o in enumerate(options):
                prompts.append(q)
                comps.append(o)
                labels.append(1 if o == gold else 0)
                comp_tok_ids.append(opt_ids[j])
            opt_ids_per_question.append(opt_ids)
    else:
        # SINGLE: for each item we will create (prompt, gold) and one synthetic negative
        # (You can customize negatives; here a simple numeric corruption or literal " 0")
        def corrupt_numeric(ans):
            s = str(ans).strip()
            try:
                v = int(s)
                return " " + str(v + 1)
            except Exception:
                return " 0"

        for it in items:
            q, gold = it["question"], it["answer"]
            prompts.append(q); comps.append(gold); labels.append(1)
            comp_tok_ids.append(first_token_id_in_context(tok, q, gold))
            neg = corrupt_numeric(gold)
            prompts.append(q); comps.append(neg);  labels.append(0)
            comp_tok_ids.append(first_token_id_in_context(tok, q, neg))

    B = len(prompts)
    print(f"[info] building hidden states for B={B} prompt/completion pairs")

    merged_texts = [_merge_prompt_completion(p, c, add_space_between=True)[0] for p, c in zip(prompts, comps)]
    layer_h = batched_lasttoken_hiddens(
        model, tok, merged_texts, batch_size=max(1, args.batch_size), skip_embedding=True
    )  # List[L] with each [B,D] on CPU

    ln_f, WU = resolve_ln_and_WU(model)
    device = next(model.parameters()).device

    # --- baseline predictions (σ=0) for flip-rate
    baseline_pred = {l: [] for l in range(L)}
    if args.task == "mcq":
        # labels per option; we also group by question later
        # we know items in order; build mapping from question idx to [option indices].
        idx = 0
        question_slices = []
        for it, opt_ids in zip(items, opt_ids_per_question):
            m = len(opt_ids)
            question_slices.append((idx, idx + m))
            idx += m
        gold_idx = []
        for it, (s, e) in zip(items, question_slices):
            y_slice = labels[s:e]
            # index of gold inside this slice
            g_local = np.argmax(y_slice)
            gold_idx.append(g_local)
    else:
        gold_idx = None  # not used for SINGLE (we use exact-match bool)

    # compute baseline logits from each layer
    for li in range(L):
        H = layer_h[li].to(device)   # [B,D] on same device as model
        z = _project_logits_from_h(H, ln_f, WU)  # [B,V]

        if args.task == "single":
            pred = torch.argmax(z, dim=-1).tolist()
            baseline_pred[li] = pred
        else:
            preds = []
            # We are using first-token logits for each option as local readout
            idx = 0
            for it, opt_ids in zip(items, opt_ids_per_question):
                m = len(opt_ids)
                # slice options for this question
                slice_logits = []
                for j in range(m):
                    tid = opt_ids[j]
                    slice_logits.append(float(z[idx + j, tid].item()))
                preds.append(int(np.argmax(slice_logits)))
                idx += m
            baseline_pred[li] = preds

    # --- per-layer std for relative noise
    sigmas = sorted(set(float(x) for x in args.sigmas.split(",") if x.strip()))
    if 0.0 not in sigmas:
        sigmas = [0.0] + sigmas

    stds = None
    if args.noise_mode == "rel":
        stds = get_layer_std_from_cache(args.out_root, model_path, args.task)
        if stds is None:
            print("[warn] fallback to abs noise (std=1)")
            args.noise_mode = "abs"

    sub = "logitlens_local"
    base_dir = Path(args.out_root) / "reports" / model_path / "robustness" / "local" / sub / args.task
    (base_dir / "figs").mkdir(parents=True, exist_ok=True)
    (base_dir / "tables").mkdir(parents=True, exist_ok=True)

    # --- sweep σ and compute acc/flip-rate; also AUC(σ,acc)
    layer_auc = []
    rows = []  # CSV rows: layer, sigma, acc, flip_rate
    device = next(model.parameters()).device

    for li in range(L):
        H = layer_h[li].to(device)      # [B,D] on same device as model
        D = H.shape[1]
        per_dim_std = None
        if args.noise_mode == "rel" and stds is not None and li in stds:
            per_dim_std = stds[li].to(device=device, dtype=torch.float32)  # [D] on model device
        else:
            per_dim_std = torch.ones(D, dtype=torch.float32, device=device)

        acc_curve, sig_curve = [], []

        for sig in sigmas:
            # deterministic seed per (layer, sigma)
            base_seed = args.seed + 1000 * li + int(sig * 1e3)
            if sig == 0.0:
                # reuse baseline
                pred = baseline_pred[li]
                preds_all = pred
            else:
                # average over repeats
                if args.task == "mcq":
                    wins = []
                corrects = 0
                preds_all = []

                for r in range(max(1, args.repeats)):
                    g_dev = torch.Generator(device=device).manual_seed(base_seed + r)
                    eps = torch.randn((B, D), device=device, generator=g_dev) * (sig * per_dim_std)
                    Hn = H + eps
                    z  = _project_logits_from_h(Hn, ln_f, WU)   # [B,V]

                    if args.task == "single":
                        pred_r = torch.argmax(z, dim=-1).tolist()
                        # compare with original labels
                        for b in range(B):
                            corrects += int(pred_r[b] == comp_tok_ids[b])
                        preds_all = pred_r
                    else:
                        idx = 0
                        preds_r = []
                        for it, opt_ids in zip(items, opt_ids_per_question):
                            m = len(opt_ids)
                            slice_logits = []
                            for j in range(m):
                                tid = opt_ids[j]
                                slice_logits.append(float(z[idx + j, tid].item()))
                            preds_r.append(int(np.argmax(slice_logits)))
                            idx += m
                        wins.append(preds_r)
                        preds_all = preds_r

                if args.task == "single":
                    acc = corrects / float(B * max(1, args.repeats))
                    pred = preds_all
                else:
                    # majority vote across repeats
                    if len(wins) > 1:
                        wins_arr = np.array(wins)  # [R, N_questions]
                        voted = []
                        for q in range(wins_arr.shape[1]):
                            vals, counts = np.unique(wins_arr[:, q], return_counts=True)
                            voted.append(int(vals[np.argmax(counts)]))
                        pred = voted
                    else:
                        pred = wins[0] if wins else baseline_pred[li]

            # accuracy and flip-rate vs baseline
            if args.task == "mcq":
                # baseline_pred[li] has shape [N_questions]
                # pred also [N_questions]; compare to gold option idx
                Nq = len(pred)
                correct_cnt = 0
                flip_cnt = 0
                for i_q, g_idx in enumerate(gold_idx):
                    if pred[i_q] == g_idx:
                        correct_cnt += 1
                    if pred[i_q] != baseline_pred[li][i_q]:
                        flip_cnt += 1
                acc  = correct_cnt / max(Nq, 1)
                flip = flip_cnt     / max(Nq, 1)
            else:
                # SINGLE: compare predicted token ids vs gold token ids
                Bcur = len(preds_all) if pred is not None else len(baseline_pred[li])
                correct_cnt = 0
                flip_cnt = 0
                for b in range(Bcur):
                    gold_tid = comp_tok_ids[b]
                    if pred[b] == gold_tid:
                        correct_cnt += 1
                    if pred[b] != baseline_pred[li][b]:
                        flip_cnt += 1
                acc  = correct_cnt / max(Bcur, 1)
                flip = flip_cnt     / max(Bcur, 1)

            acc_curve.append(acc)
            sig_curve.append(sig)
            rows.append({
                "layer": li,
                "sigma": sig,
                "acc": acc,
                "flip_rate": flip,
            })

        # AUC(σ,acc) for this layer
        auc_val = normalized_auc(sig_curve, acc_curve)
        layer_auc.append({"layer": li, "robustness_auc": auc_val})

        # plot acc vs sigma
        plt.figure(figsize=(5, 4))
        order = np.argsort(sig_curve)
        xs = np.array(sig_curve)[order]
        ys = np.array(acc_curve)[order]
        plt.plot(xs, ys, marker="o")
        plt.xlabel("σ (noise std)")
        plt.ylabel("Accuracy vs baseline")
        plt.title(f"Layer {li}: ACC vs σ ({args.task})")
        plt.grid(alpha=0.3)
        out_png = base_dir / "figs" / f"{args.task}_acc_vs_sigma_layer{li}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[SAVE] {out_png}")

    # big CSV of (layer, sigma, acc, flip_rate)
    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = base_dir / "tables" / f"{args.task}_local_logitlens_noise_curves.csv"
    df.to_csv(out_csv, index=False)
    print("[SAVE]", out_csv)

    # robustness per layer
    df_auc = pd.DataFrame(layer_auc).sort_values("layer")
    out_csv2 = base_dir / "tables" / f"{args.task}_local_logitlens_robustness_per_layer.csv"
    df_auc.to_csv(out_csv2, index=False)
    print("[SAVE]", out_csv2)

    # plot robustness vs layer
    plt.figure(figsize=(6, 4))
    plt.plot(df_auc["layer"], df_auc["robustness_auc"], marker="s")
    plt.xlabel("Layer")
    plt.ylabel("Robustness AUC (ACC–σ)")
    plt.title(f"Local logit-lens robustness per layer ({args.task})")
    plt.grid(alpha=0.3)
    out_png2 = base_dir / "figs" / f"{args.task}_local_logitlens_robustness_vs_layer.png"
    plt.tight_layout()
    plt.savefig(out_png2, dpi=150)
    plt.close()
    print("[SAVE]", out_png2)


if __name__ == "__main__":
    main()
