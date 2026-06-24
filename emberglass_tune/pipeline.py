#!/usr/bin/env python3
"""Unified verify -> smoke -> train -> merge pipeline with hardware presets."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from emberglass_tune.presets import load_preset


def _run(cmd: list[str]) -> None:
    print(f"[pipeline] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _train_args(section: dict, overrides: dict) -> list[str]:
    keys = {
        "epochs": "--epochs",
        "max_steps": "--max-steps",
        "bs": "--bs",
        "grad_accum": "--grad-accum",
        "lr": "--lr",
        "max_len": "--max-len",
        "save_steps": "--save-steps",
        "valid_frac": "--valid-frac",
        "limit": "--limit",
        "lora_r": "--lora-r",
        "lora_alpha": "--lora-alpha",
        "seed": "--seed",
    }
    flags = {
        "bucket_batch": "--bucket-batch",
        "no_bucket_batch": "--no-bucket-batch",
    }
    out: list[str] = []
    merged = {**section, **{k: v for k, v in overrides.items() if v is not None}}
    for key, flag in keys.items():
        if key in merged and merged[key] is not None:
            out.extend([flag, str(merged[key])])
    for key, flag in flags.items():
        if merged.get(key):
            out.append(flag)
    if merged.get("attn"):
        out.extend(["--attn", str(merged["attn"])])
    if merged.get("cache_dir"):
        out.extend(["--cache-dir", str(merged["cache_dir"])])
    if merged.get("dataloader_workers") is not None:
        out.extend(["--dataloader-workers", str(merged["dataloader_workers"])])
    if merged.get("eval_steps") is not None:
        out.extend(["--eval-steps", str(merged["eval_steps"])])
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Run verify -> smoke -> train -> merge using a hardware preset.",
    )
    ap.add_argument("--preset", default="lambda-gh200", help="configs/presets/<name>.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="Final adapter output dir")
    ap.add_argument("--valid", default="", help="Optional held-out JSONL")
    ap.add_argument("--merged-out", default="", help="Merged model dir (default: skip merge)")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-merge", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args, _ = ap.parse_known_args(argv)

    preset = load_preset(args.preset)
    cli = ["emberglass-tune"]
    merged_out = args.merged_out or preset.get("merged_out", "")

    verify = preset.get("verify", {})
    smoke = preset.get("smoke", {})
    train = preset.get("train", {})
    do_merge = preset.get("merge", True) and not args.skip_merge
    run_full_train = preset.get("run_full_train", True) and not args.skip_train

    steps: list[list[str]] = []

    if not args.skip_verify:
        steps.append(
            cli
            + [
                "verify",
                "--model",
                args.model,
                "--data",
                args.data,
                "--min-usable",
                str(verify.get("min_usable", 100)),
                "--max-len",
                str(verify.get("max_len", train.get("max_len", 8192))),
            ]
        )

    if not args.skip_smoke:
        smoke_out = smoke.get("out", "adapters/_smoke")
        cmd = (
            cli
            + ["train", "--model", args.model, "--data", args.data, "--out", smoke_out]
            + _train_args(smoke, {})
        )
        if args.valid:
            cmd.extend(["--valid", args.valid])
        steps.append(cmd)

    if run_full_train:
        train_cmd = (
            cli + ["train", "--model", args.model, "--data", args.data, "--out", args.out]
            + _train_args(train, {})
        )
        if args.valid:
            train_cmd.extend(["--valid", args.valid])
        steps.append(train_cmd)

    if do_merge and merged_out:
        steps.append(
            cli
            + ["merge", "--base", args.model, "--adapter", args.out, "--out", merged_out]
        )

    print(
        f"[pipeline] preset={args.preset} steps={len(steps)} "
        f"model={args.model} data={args.data}",
        flush=True,
    )

    if args.dry_run:
        for i, cmd in enumerate(steps, 1):
            print(f"[pipeline] step {i}/{len(steps)}: {' '.join(cmd)}", flush=True)
        return

    for i, cmd in enumerate(steps, 1):
        print(f"[pipeline] step {i}/{len(steps)}", flush=True)
        _run(cmd)

    print(f"[pipeline] DONE adapter={args.out}", flush=True)
    if do_merge and merged_out:
        print(f"[pipeline] merged={merged_out}", flush=True)


if __name__ == "__main__":
    main()
