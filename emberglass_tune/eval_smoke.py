#!/usr/bin/env python3
"""Generic CUDA/PEFT smoke eval: load base+adapter, one greedy generation."""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="CUDA smoke eval for a LoRA adapter")
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompt", default="Reply with one word: ok")
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval-smoke] device={device} base={args.base} adapter={args.adapter}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval().to(device)

    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False)
    reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"[eval-smoke] reply={reply!r}", flush=True)
    print("[eval-smoke] OK", flush=True)


if __name__ == "__main__":
    main()
