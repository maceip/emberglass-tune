#!/usr/bin/env python3
"""emberglass-tune CLI — train, merge, verify, trace generation."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="emberglass-tune",
        description="LoRA / SFT training for Qwen2.5-class models. "
        "Browser inference lives in the emberglass (qwen-webgpu-lora) repo.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="PEFT LoRA SFT (GPU / CPU)")
    p_train.set_defaults(_run="train")

    p_merge = sub.add_parser("merge", help="Merge adapter into base weights")
    p_merge.set_defaults(_run="merge")

    p_verify = sub.add_parser("verify", help="Tokenization preflight on trace JSONL")
    p_verify.set_defaults(_run="verify")

    p_gate = sub.add_parser("gate-traces", help="Fail-closed trace quality gate")
    p_gate.set_defaults(_run="gate")

    p_traces = sub.add_parser("traces", help="Generate teacher reasoning traces")
    p_traces.set_defaults(_run="traces")

    p_eval = sub.add_parser("eval-smoke", help="CUDA/CPU one-shot generation smoke test")
    p_eval.set_defaults(_run="eval_smoke")

    p_pipe = sub.add_parser(
        "pipeline",
        help="Verify -> smoke -> train -> merge using a hardware preset",
    )
    p_pipe.set_defaults(_run="pipeline")

    args, rest = parser.parse_known_args(argv)

    if args._run == "train":
        from emberglass_tune.train_sft import main as run

        sys.argv = ["train_sft", *rest]
        run()
    elif args._run == "merge":
        from emberglass_tune.merge_lora import main as run

        sys.argv = ["merge_lora", *rest]
        run()
    elif args._run == "verify":
        from emberglass_tune.verify_sft_data import main as run

        sys.argv = ["verify_sft_data", *rest]
        run()
    elif args._run == "gate":
        from emberglass_tune.trace_gate import main as run

        sys.argv = ["trace_gate", *rest]
        run()
    elif args._run == "traces":
        from emberglass_tune.trace_gen import main as run

        sys.argv = ["trace_gen", *rest]
        run()
    elif args._run == "eval_smoke":
        from emberglass_tune.eval_smoke import main as run

        sys.argv = ["eval_smoke", *rest]
        run()
    elif args._run == "pipeline":
        from emberglass_tune.pipeline import main as run

        run(rest)
    else:
        parser.error(f"unknown command {args._run}")


if __name__ == "__main__":
    main()
