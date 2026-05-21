#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build prompt_pool.json for a given model.

Tasks generated:
  - MCQ (math):    "x + y = " with 4 numeric *single-token* options (under this model's tokenizer).
                   Prompt includes: "Choose exactly one option from the list."
  - Single (math): "x + y = " with *single-token* gold numeric answer (under this model's tokenizer).
                   Prompt includes: "Respond with exactly one token (the number)."
  - MCQ (capital True/False): "<Capital> is the capital of <Country>." with options [True/False-like tokens].
                   Prompt includes: "Choose exactly one option from the list."
  - Single (capital True/False): same statement.
                   Prompt includes: "Respond with exactly one token: True or False."

IMPORTANT:
  - Single-token constraints are enforced per model by checking tokenizer.encode(...).
  - Math items that violate the constraint are discarded and re-sampled until the requested count is reached.
"""

import json
import random
import argparse
import sys
from pathlib import Path
import os

# make 'src' importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.util import load_model_and_tokenizer, get_device  # type: ignore


# -------------------------------------------------------------------------
# Countries: (country, correct_capital, wrong_cap_1, wrong_cap_2)
# We will convert these into True/False statements:
#   "<capital> is the capital of <country>."
# -------------------------------------------------------------------------
COUNTRIES = [
    ("France",  "Paris",   "London",         "Berlin"),
    ("Germany", "Berlin",  "Paris",          "Rome"),
    ("Italy",   "Rome",    "Madrid",         "Berlin"),
    ("Spain",   "Madrid",  "Rome",           "Lisbon"),
    ("Japan",   "Tokyo",   "Osaka",          "Kyoto"),
    ("Canada",  "Ottawa",  "Toronto",        "Montreal"),
    ("Brazil",  "Brasilia","Rio",            "Sao Paulo"),
    ("India",   "New Delhi","Mumbai",        "Kolkata"),
    ("China",   "Beijing", "Shanghai",       "Shenzhen"),
    ("Australia","Canberra","Sydney",        "Melbourne"),
    ("Iran",    "Tehran",  "Isfahan",        "Shiraz"),
    ("Turkey",  "Ankara",  "Istanbul",       "Izmir"),
    ("Russia",  "Moscow",  "Saint Petersburg","Kazan"),
    ("Egypt",   "Cairo",   "Alexandria",     "Giza"),
    ("Mexico",  "Mexico City","Guadalajara", "Monterrey"),
]


# -------------------------------------------------------------------------
# Helper: check if a string is single-token under this tokenizer
# -------------------------------------------------------------------------
def is_single_token(tokenizer, s: str) -> bool:
    s = s.strip()
    ids = tokenizer.encode(s, add_special_tokens=False)
    return len(ids) == 1


def all_single_token(tokenizer, strings) -> bool:
    return all(is_single_token(tokenizer, s) for s in strings)


# -------------------------------------------------------------------------
# Choose label tokens for True/False that are single-token if possible
# Candidates are tried in order; first pair that is single-token wins.
# -------------------------------------------------------------------------
TF_CANDIDATES = [
    (" True",  " False"),
    (" yes",   " no"),
    (" Yes",   " No"),
]


def choose_tf_tokens(tokenizer):
    """
    Return (token_for_true, token_for_false) such that each is (ideally)
    single-token for this tokenizer.
    """
    for t_pos, t_neg in TF_CANDIDATES:
        if is_single_token(tokenizer, t_pos) and is_single_token(tokenizer, t_neg):
            return t_pos, t_neg
    # Fallback: use the last pair, even if not strictly single-token.
    return TF_CANDIDATES[-1]


# -------------------------------------------------------------------------
# Prompt instructions (task-type aware)
# -------------------------------------------------------------------------
INSTR_MCq_CHOOSE_ONE = " (Choose exactly one option from the list.) "
INSTR_TF_SINGLE_ONE_TOKEN = " (Respond with exactly one token: True or False.) "
INSTR_MATH_SINGLE_ONE_TOKEN = " (Respond with exactly one number token.) "


# -------------------------------------------------------------------------
# Main builder
# -------------------------------------------------------------------------
def build_prompt_pool(
    tokenizer,
    n_math: int = 400,
    cap_reps: int = 20,
    seed: int = 0,
    max_tries_multiplier: int = 500,
):
    """
    Build a mixed pool of MCQ + Single items:
      - True/False capital statements
      - numeric addition questions (strict single-token answers/options)

    Returns:
        items: list[dict] ready to be dumped as JSON.

    Notes:
      - Math items are generated with rejection sampling to enforce single-token constraints.
      - If constraints are too strict for a tokenizer, the generator may fail after many tries.
    """
    random.seed(seed)
    items = []

    # ----- choose label tokens (True / False) -----
    tf_true, tf_false = choose_tf_tokens(tokenizer)

    # ------------------------------------------------------------------
    # 1) Capital True/False (both MCQ and Single)
    # ------------------------------------------------------------------
    for c, a, b, d in COUNTRIES * cap_reps:
        a = a.strip()
        b = b.strip()
        d = d.strip()

        # True statement
        base_true = f"{a} is the capital of {c}."
        q_true_mcq = base_true + INSTR_MCq_CHOOSE_ONE
        q_true_single = base_true + INSTR_TF_SINGLE_ONE_TOKEN

        items.append({
            "id": f"mcq-cap-tf-true-{c.lower()}-{random.randint(0, 999999)}",
            "task": "mcq",
            "question": q_true_mcq,
            "options": [tf_true, tf_false],
            "answer": tf_true,
        })
        items.append({
            "id": f"single-cap-tf-true-{c.lower()}-{random.randint(0, 999999)}",
            "task": "single",
            "question": q_true_single,
            "answer": tf_true,
        })

        # False statements using the wrong capitals (b and d)
        for wrong_cap in (b, d):
            wrong_cap = wrong_cap.strip()
            base_false = f"{wrong_cap} is the capital of {c}."
            q_false_mcq = base_false + INSTR_MCq_CHOOSE_ONE
            q_false_single = base_false + INSTR_TF_SINGLE_ONE_TOKEN

            items.append({
                "id": f"mcq-cap-tf-false-{c.lower()}-{random.randint(0, 999999)}",
                "task": "mcq",
                "question": q_false_mcq,
                "options": [tf_true, tf_false],
                "answer": tf_false,
            })
            items.append({
                "id": f"single-cap-tf-false-{c.lower()}-{random.randint(0, 999999)}",
                "task": "single",
                "question": q_false_single,
                "answer": tf_false,
            })

    # ------------------------------------------------------------------
    # 2) Simple math additions (MCQ) with strict single-token options
    # ------------------------------------------------------------------
    mcq_math = []
    tries = 0
    max_tries = max(1, n_math) * max_tries_multiplier

    while len(mcq_math) < n_math and tries < max_tries:
        tries += 1
        x = random.randint(2, 60)
        y = random.randint(2, 60)

        #gold = f" {x + y}"
        gold = str(x + y)
        #decoys = [f" {x + y + 1}", f" {x + y - 1}", f" {x + y + 2}"]
        decoys = [str(x + y + 1), str(x + y - 1), str(x + y + 2)]
        opts = [gold] + decoys

        # Enforce: ALL options must be single-token for this tokenizer.
        if not all_single_token(tokenizer, opts):
            continue

        random.shuffle(opts)
        mcq_math.append({
            "id": f"mcq-math-{x}-{y}-{random.randint(0, 999999)}",
            "task": "mcq",
            "question": f"{x} + {y} = " + INSTR_MCq_CHOOSE_ONE,
            "options": opts,
            "answer": gold,
        })

    if len(mcq_math) < n_math:
        raise RuntimeError(
            f"Failed to generate enough single-token MCQ math items. "
            f"Requested={n_math}, generated={len(mcq_math)}, tries={tries}, max_tries={max_tries}. "
            f"Consider increasing max_tries_multiplier, narrowing number ranges, or changing decoy strategy."
        )

    items.extend(mcq_math)

    # ------------------------------------------------------------------
    # 3) Simple math additions (Single) with strict single-token gold
    # ------------------------------------------------------------------
    single_math = []
    tries = 0
    max_tries = max(1, n_math) * max_tries_multiplier

    while len(single_math) < n_math and tries < max_tries:
        tries += 1
        x = random.randint(2, 60)
        y = random.randint(2, 60)

        #gold = f" {x + y}"
        gold = str(x + y)

        # Enforce: gold must be single-token for this tokenizer.
        if not is_single_token(tokenizer, gold):
            continue

        single_math.append({
            "id": f"single-math-{x}-{y}-{random.randint(0, 999999)}",
            "task": "single",
            "question": f"{x} + {y} = " + INSTR_MATH_SINGLE_ONE_TOKEN,
            "answer": gold,
        })

    if len(single_math) < n_math:
        raise RuntimeError(
            f"Failed to generate enough single-token SINGLE math items. "
            f"Requested={n_math}, generated={len(single_math)}, tries={tries}, max_tries={max_tries}. "
            f"Consider increasing max_tries_multiplier or narrowing number ranges."
        )

    items.extend(single_math)

    random.shuffle(items)
    return items


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build prompt_pool.json for MCQ + Single tasks.")
    parser.add_argument("--model", default="EleutherAI/gpt-neo-125M",
                        help="HF model name, e.g. EleutherAI/gpt-neo-125M")
    parser.add_argument("--n_math", type=int, default=400,
                        help="Number of math questions (MCQ + Single)")
    parser.add_argument("--cap_reps", type=int, default=20,
                        help="How many times to repeat the COUNTRY list to enlarge TF data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_tries_multiplier", type=int, default=500,
                        help="Max tries = n_math * max_tries_multiplier (for rejection sampling)")
    args = parser.parse_args()
    #os.environ["HF_TOKEN"] = "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"
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    items = build_prompt_pool(
        tokenizer,
        n_math=args.n_math,
        cap_reps=args.cap_reps,
        seed=args.seed,
        max_tries_multiplier=args.max_tries_multiplier,
    )

    # Save under data/<model_path>/prompt_pool.json
    model_path = args.model.replace("/", "__")
    out_dir = REPO_ROOT / "data" / model_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "prompt_pool.json"

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Saved prompt_pool.json with {len(items)} items at {out_file}")