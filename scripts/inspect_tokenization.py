"""
Inspect how a single example is tokenized and what the train/eval loops see.

Usage:
    python scripts/inspect_tokenization.py
    python scripts/inspect_tokenization.py --task amazon_polarity --index 5
    python scripts/inspect_tokenization.py --task dbpedia_14 --max_length 320
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from src.data.datasets import DATASET_REGISTRY, _format_example


def color(text, code):
    return f"\033[{code}m{text}\033[0m"

def green(t):  return color(t, "32")
def red(t):    return color(t, "31")
def yellow(t): return color(t, "33")
def cyan(t):   return color(t, "36")
def bold(t):   return color(t, "1")


def inspect(task_name: str, sample_index: int, max_length: int, model_name: str):
    cfg = DATASET_REGISTRY[task_name]

    from datasets import load_dataset
    ds = load_dataset(cfg["path"])
    raw = ds["train"][sample_index]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    result = _format_example(
        raw,
        text_fields=cfg["text_fields"],
        label_field=cfg["label_field"],
        label_names=cfg["label_names"],
        instruction=cfg["instruction"],
        tokenizer=tokenizer,
        max_length=max_length,
    )

    input_ids      = result["input_ids"]
    attention_mask = result["attention_mask"]
    labels         = result["labels"]
    label_idx      = result["label_idx"]

    # ── Raw example ──────────────────────────────────────────────────────────
    print(bold("\n" + "="*70))
    print(bold(f"  TASK: {task_name}   SAMPLE INDEX: {sample_index}"))
    print(bold("="*70))

    print(cyan("\n── Raw fields ──────────────────────────────────────────────────────"))
    for field in cfg["text_fields"]:
        val = str(raw.get(field, ""))
        print(f"  {field}: {val[:200]}{'...' if len(val) > 200 else ''}")
    print(f"  label index : {label_idx}")
    print(f"  label string: {cfg['label_names'][label_idx]}")

    # ── Reconstructed prompt ──────────────────────────────────────────────────
    options_str = ", ".join(cfg["label_names"])
    full_decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(cyan("\n── Full decoded sequence (what the model sees) ─────────────────────"))
    print(f"  {repr(full_decoded)}")

    # ── Token-by-token table ──────────────────────────────────────────────────
    print(cyan("\n── Token-by-token breakdown ────────────────────────────────────────"))
    print(f"  {'pos':>4}  {'token_id':>8}  {'decoded':>20}  {'label':>10}  {'attn':>4}  role")
    print(f"  {'---':>4}  {'--------':>8}  {'-------':>20}  {'-----':>10}  {'----':>4}  ----")

    real_tokens = sum(1 for m in attention_mask if m == 1)
    pad_tokens  = len(input_ids) - real_tokens

    target_positions = [i for i, l in enumerate(labels) if l != -100 and attention_mask[i] == 1]
    pred_pos = target_positions[0] - 1 if target_positions else None

    for pos, (tid, lbl, msk) in enumerate(zip(input_ids, labels, attention_mask)):
        decoded = repr(tokenizer.decode([tid], skip_special_tokens=False))

        if msk == 0:
            role = red("PAD")
        elif lbl != -100:
            role = green("LABEL ← train loss here")
        elif pos == pred_pos:
            role = yellow("PRED_POS ← eval reads logit here")
        elif pos == 0:
            role = "BOS"
        else:
            role = "prompt"

        lbl_str = str(lbl) if lbl != -100 else "-100"
        print(f"  {pos:>4}  {tid:>8}  {decoded:>20}  {lbl_str:>10}  {msk:>4}  {role}")

        if msk == 0 and pos == real_tokens:
            print(f"  ... ({pad_tokens} padding tokens) ...")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    print(cyan("\n── Summary ─────────────────────────────────────────────────────────"))
    print(f"  max_length           : {max_length}")
    print(f"  real tokens (non-pad): {real_tokens}")
    print(f"  padding tokens       : {pad_tokens}")

    if target_positions:
        label_token_id = input_ids[target_positions[0]]
        label_decoded  = tokenizer.decode([label_token_id], skip_special_tokens=True)
        print(f"  label token position : {target_positions[0]}")
        print(f"  label token id       : {label_token_id}  →  {repr(label_decoded)}")
        print(f"  eval pred_pos        : {pred_pos}  (logit here predicts position {target_positions[0]})")
    else:
        print(red("  WARNING: no label token found — labels are all -100!"))

    # ── Eval simulation ───────────────────────────────────────────────────────
    print(cyan("\n── Eval simulation (what the eval loop does) ───────────────────────"))
    label_token_ids = []
    for name in cfg["label_names"]:
        toks = tokenizer.encode(f" {name}", add_special_tokens=False)
        label_token_ids.append(toks[0])
        decoded_name = tokenizer.decode([toks[0]])
        print(f"  label '{name}' → token_id {toks[0]} → decoded: {repr(decoded_name)}")

    print(f"\n  At eval, model reads logit at pos {pred_pos}")
    print(f"  Restricts to token_ids: {label_token_ids}")
    print(f"  Argmax over those → compare to true label_idx={label_idx} ('{cfg['label_names'][label_idx]}')")

    # ── Instruction structure ─────────────────────────────────────────────────
    print(cyan("\n── Instruction structure ────────────────────────────────────────────"))
    print(f"  instruction: {cfg['instruction']}")
    print(f"  options    : {options_str}")
    print(bold("\n" + "="*70 + "\n"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       default="ag_news",
                        choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--index",      type=int, default=0,
                        help="Sample index in the train split")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--model",      default="Qwen/Qwen2.5-1.5B")
    args = parser.parse_args()

    inspect(args.task, args.index, args.max_length, args.model)


if __name__ == "__main__":
    main()
