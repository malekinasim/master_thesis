
from typing import Optional, Dict, Any, List
import torch
from src.tuned import TunedDiag
from src.logit_lens import *
from typing import Dict
import numpy as np
from typing import List, Dict, Union, Sequence
import torch

def _to_str(x):
    # convert numpy scalars / arrays and bytes to clean str
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return str(x)
    if isinstance(x, (np.generic, np.ndarray)):
        try:
            # scalar array
            if np.ndim(x) == 0:
                x = x.item()
            else:
                # join vector elements (fallback)
                x = " ".join(map(str, np.ravel(x).tolist()))
        except Exception:
            x = str(x)
    return str(x)

@torch.no_grad()
def layerwise_logits_for_pos(
    model, tokenizer, text=None, outputs=None, pos=-1,
    ln_f_mode="last_only",      # "none" | "last_only" | "all"
    skip_embedding=False,       # True => start from block1 (drop embedding row)
    tuned=None,                 # TunedDiag or compatible
    option_ids=None             # if provided -> return logits only for these ids
):
    """
    Returns:
      - if tuned is None: list[Tensor]   (per-layer logits)
      - else: {"raw": list[Tensor], "tuned": list[Tensor]}
    If option_ids is not None, each Tensor has shape [|options|] instead of [V].
    """
    device = next(model.parameters()).device

    # forward (or reuse)
    if outputs is None:
        assert text is not None, "Provide `text` or `outputs`"
        inputs = tokenizer(str(text), return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    hs = outputs.hidden_states                 # [emb, h1, ..., hL]
    assert isinstance(hs, (tuple, list)) and len(hs) >= 2, "hidden_states missing"

    # resolve readout matrix (W_U) safely
    lm_head = getattr(model, "lm_head", None)
    assert lm_head is not None and hasattr(lm_head, "weight"), "model.lm_head.weight missing"
    W_U = lm_head.weight.T.contiguous()       # [d, V]

    # resolve final LayerNorm in a model-agnostic way
    def _resolve_attr(obj, path):
        cur = obj
        for part in path.split("."):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur
    ln_f = None
    for cand in ("transformer.ln_f", "model.norm", "model.decoder.norm"):
        ln_f = _resolve_attr(model, cand)
        if ln_f is not None:
            break
    if ln_f_mode not in {"none", "last_only", "all"}:
        ln_f_mode = "last_only"

    start = 1 if skip_embedding else 0
    L_total = len(hs)
    L_last = L_total - 1

    # restrict to selected option columns (if provided)
    WU_opts = None
    if option_ids is not None:
        opt_idx = torch.as_tensor(option_ids, device=W_U.device, dtype=torch.long)
        WU_opts = torch.index_select(W_U, dim=1, index=opt_idx)  # [d, |opts|]

    # helper: apply ln_f (if requested) then project
    def project_vec(x, i):
        if ln_f is not None and ln_f_mode != "none":
            if (ln_f_mode == "last_only" and i == L_last) or (ln_f_mode == "all"):
                x = ln_f(x)
        return (x @ (WU_opts if WU_opts is not None else W_U))

    # choose position robustly (allow negative indexing like -1)
    # hs[i] shape: [B=1, T, D]
    T = hs[-1].shape[1]
    pos_eff = pos if pos >= 0 else (T + pos)
    pos_eff = max(0, min(T - 1, pos_eff))

    layer_raw, layer_tuned = [], []
    for i in range(start, L_total):
        x = hs[i][0, pos_eff]                  # [d]
        # raw logits
        z_raw = project_vec(x, i)
        layer_raw.append(z_raw)

        # tuned logits (support both apply_x and legacy apply; scale by alpha if present)
        if tuned is not None:
            if hasattr(tuned, "apply_x"):
                xt = tuned.apply_x(i, x)
            else:
                xt = tuned.apply(i, x)   # legacy fallback

            z_tuned = project_vec(xt, i)
            a_i = tuned.alpha(i) if hasattr(tuned, "alpha") else None
            if a_i is not None:
                z_tuned = float(a_i) * z_tuned
            layer_tuned.append(z_tuned)

    return layer_raw if tuned is None else {"raw": layer_raw, "tuned": layer_tuned}


@torch.no_grad()
def _alllayer_lasttoken_hiddens_core(
    model,
    tokenizer,
    merged_texts: List[str],
    skip_embedding: bool = True,
):
    """
    Core routine: given a batch of full sequences (already merged 'prompt + completion'),
    return per-layer hidden vectors at the *last non-pad token* for each sequence.
    Returns:
        layer_h: List[Tensor], length = n_layers(± embedding), each [B, D]
    """
    device = next(model.parameters()).device

    # ensure pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        merged_texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    out = model(**enc, output_hidden_states=True, use_cache=False)
    if not hasattr(out, "hidden_states") or out.hidden_states is None:
        raise RuntimeError("model outputs missing hidden_states; ensure output_hidden_states=True")

    hs = out.hidden_states  # [emb, h1, ..., hL]

    start = 1 if skip_embedding else 0
    T = enc["input_ids"].shape[1]

    ids = enc["input_ids"]
    ar = torch.arange(T, device=ids.device).unsqueeze(0).expand_as(ids)
    mask = (ids != tokenizer.pad_token_id)
    last_idx = (ar * mask).max(dim=1).values  # [B]

    layer_h = []
    for i in range(start, len(hs)):
        H = hs[i]                  # [B, T, D]
        B, _, D = H.shape
        vecs = H[torch.arange(B, device=H.device), last_idx]  # [B, D]
        layer_h.append(vecs.detach().cpu())
    return layer_h  # length = n_layers (± embedding), each [B, D]



# --- helper: محاسبه‌ی اندیس توکن انتخابی برای هر آیتم بچ ---
def _calc_pos_indices_from_spec(
    enc,                       # خروجی tokenizer(..., return_tensors='pt', padding=True, truncation=True, [return_offsets_mapping])
    pos_spec: Union[int, Sequence[int], Dict, None],
    pad_id: int,
    comp_starts_char: Sequence[int] | None = None,   # طول کاراکتری بخش prompt (+space) برای هر آیتم
):
    ids = enc["input_ids"]                  # [B, T]
    am  = enc.get("attention_mask", None)   # [B, T] یا None
    device = ids.device
    B, T = ids.shape

    if am is None:
        am = torch.ones_like(ids, dtype=torch.long)

    lens = am.sum(dim=1)                    # [B] تعداد توکن‌های غیر-PAD هر سکانس

    # پیش‌فرض/رفتار قبلی: آخرین توکن غیر-PAD
    if pos_spec in (None, -1, "last", "last_nonpad"):
        return lens - 1

    # حالت int سراسری    
    if isinstance(pos_spec, int):
        pos = torch.full((B,), pos_spec, device=device, dtype=torch.long)
        pos = torch.where(pos >= 0, pos, lens + pos)
        # --- پچ: clamp امن ---
        max_allowed = torch.clamp_min(lens - 1, 0)
        pos = torch.clamp(pos, min=0)
        pos = torch.minimum(pos, max_allowed)
        return pos

    # حالت لیست برای هر آیتم
    if isinstance(pos_spec, (list, tuple)):
        if len(pos_spec) != B:
            raise ValueError("pos_spec list length must match batch size.")
        pos = torch.as_tensor(pos_spec, device=device, dtype=torch.long)
        pos = torch.where(pos >= 0, pos, lens + pos)
        # --- پچ: clamp امن ---
        max_allowed = torch.clamp_min(lens - 1, 0)
        pos = torch.clamp(pos, min=0)
        pos = torch.minimum(pos, max_allowed)
        return pos

    # حالت نسبی به completion
    if isinstance(pos_spec, dict) and pos_spec.get("mode") == "comp":
        if "offset_mapping" not in enc:
            raise ValueError("pos_spec={'mode':'comp',...} requires return_offsets_mapping=True in tokenizer()")
        if comp_starts_char is None:
            raise ValueError("comp_starts_char required for mode='comp' (char start of completion per example).")

        off = enc["offset_mapping"]         # [B, T, 2]، شروع/پایان کاراکتری هر توکن
        k   = int(pos_spec.get("k", 0))
        idx = torch.zeros(B, dtype=torch.long, device=device)

        for b in range(B):
            valid     = am[b].bool()
            start_chr = int(comp_starts_char[b])
            offs      = off[b]                               # [T,2]
            # توکن‌های مربوط به completion: آنهایی که شروع‌شان >= start_chr (و غیر-PAD)
            comp_mask = (offs[:, 1] > start_chr) & (offs[:, 0] >= start_chr)
            cand      = torch.nonzero(valid & comp_mask, as_tuple=False).squeeze(-1)
            if cand.numel() == 0:
                # اگر به هر دلیلی تطابق نبود، برگرد به آخرین غیر-PAD
                idx[b] = lens[b] - 1
            else:
                kk = k if k >= 0 else cand.numel() + k
                kk = min(max(0, kk), cand.numel() - 1)
                idx[b] = cand[kk]
        return idx

    raise ValueError(f"Unsupported pos_spec: {pos_spec}")
# --- هستهٔ جدید: بردارهای لایه‌ای برای توکن انتخابی ---
@torch.no_grad()
def _alllayer_tokenpos_hiddens_core(
    model,
    tokenizer,
    merged_texts: List[str],
    pos_spec: Union[int, Sequence[int], Dict, None] = -1,
    skip_embedding: bool = True,
    comp_starts_char: Sequence[int] | None = None,   # برای mode='comp'
    need_offsets: bool = False,                      # اگر mode='comp' است True
):
    device = next(model.parameters()).device

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        merged_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=need_offsets
    ).to(device)

    out = model(**enc, output_hidden_states=True, use_cache=False)
    if not hasattr(out, "hidden_states") or out.hidden_states is None:
        raise RuntimeError("model outputs missing hidden_states; ensure output_hidden_states=True")

    hs = out.hidden_states  # [emb, h1, ..., hL]
    start = 1 if skip_embedding else 0

    pos_idx = _calc_pos_indices_from_spec(
        enc,
        pos_spec=pos_spec,
        pad_id=tokenizer.pad_token_id,
        comp_starts_char=comp_starts_char
    )  # [B]

    layer_h = []
    for i in range(start, len(hs)):
        H = hs[i]  # [B, T, D]
        B = H.shape[0]
        vecs = H[torch.arange(B, device=H.device), pos_idx]  # [B, D]
        layer_h.append(vecs.detach().cpu())
    return layer_h

@torch.no_grad()
def single_alllayer_hiddens(
    model,
    tokenizer,
    prompt_texts: List[str],
    completions: List[str],
    skip_embedding: bool = True,
    add_space_between: bool = True,
    pos_spec: Union[int, Sequence[int], Dict, None] = -1,  # NEW
):
    merged = []
    comp_starts_char = []
    for p, c in zip(prompt_texts, completions):
        p = _to_str(p)
        c = _to_str(c)
        if add_space_between and c and not c[:1].isspace():
            merged.append(p + " " + c)
            comp_starts_char.append(len(p) + 1)
        else:
            merged.append(p + c)
            comp_starts_char.append(len(p))

    need_offsets = isinstance(pos_spec, dict) and pos_spec.get("mode") == "comp"
    return _alllayer_tokenpos_hiddens_core(
        model, tokenizer, merged,
        pos_spec=pos_spec,
        skip_embedding=skip_embedding,
        comp_starts_char=comp_starts_char,
        need_offsets=need_offsets
    )


@torch.no_grad()
def mcq_alllayer_hiddens(
    model,
    tokenizer,
    prompt_text: str,
    options: List[str],
    skip_embedding: bool = True,
    add_space_between: bool = True,
    pos_spec: Union[int, Sequence[int], Dict, None] = -1,  # NEW
) -> List[Dict[str, torch.Tensor]]:
    prompt_texts = [prompt_text] * len(options)
    layer_h = single_alllayer_hiddens(
        model, tokenizer, prompt_texts, options,
        skip_embedding=skip_embedding,
        add_space_between=add_space_between,
        pos_spec=pos_spec
    )
    layer_dicts: List[Dict[str, torch.Tensor]] = []
    for layer_i, mat in enumerate(layer_h):
        d = {opt: mat[b] for b, opt in enumerate(options)}  # mat: [B, D]
        layer_dicts.append(d)
    return layer_dicts
