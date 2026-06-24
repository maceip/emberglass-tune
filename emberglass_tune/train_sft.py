#!/usr/bin/env python3
"""Phase-1 cold-start SFT: LoRA fine-tune VibeThinker-3B on faithful reasoning traces."""
import argparse
import sys

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from emberglass_tune.tokenize import (
    IGNORE,
    LengthGroupedBatchSampler,
    load_or_build_examples,
    read_jsonl,
    resolve_attn,
)


class BucketTrainer(Trainer):
    """Trainer that batches by similar sequence length to reduce padding."""

    def __init__(self, *args, bucket_batch: bool = False, **kwargs):
        self.bucket_batch = bucket_batch
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if not self.bucket_batch:
            return super().get_train_dataloader()
        from torch.utils.data import DataLoader

        lengths = [len(x["input_ids"]) for x in self.train_dataset]
        batch_sampler = LengthGroupedBatchSampler(
            lengths,
            batch_size=self.args.per_device_train_batch_size,
            world_size=max(1, self.args.world_size),
            seed=self.args.seed,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--valid", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--valid-frac", type=float, default=0.04)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=0, help="0 = same as save-steps")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--attn",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
        help="Attention backend (auto tries flash_attn when installed)",
    )
    ap.add_argument(
        "--bucket-batch",
        action="store_true",
        default=True,
        help="Batch similar-length sequences (default: on)",
    )
    ap.add_argument("--no-bucket-batch", action="store_true")
    ap.add_argument(
        "--cache-dir",
        default="",
        help="Reuse tokenized examples on disk (.emberglass-cache)",
    )
    ap.add_argument("--dataloader-workers", type=int, default=2)
    args = ap.parse_args()

    bucket_batch = args.bucket_batch and not args.no_bucket_batch
    eval_steps = args.eval_steps or args.save_steps
    attn = resolve_attn(args.attn)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[sft] loaded {len(rows)} trace rows", flush=True)

    cache_dir = args.cache_dir or None
    examples, dropped = load_or_build_examples(
        rows,
        tok,
        args.max_len,
        cache_dir,
        args.model,
        args.data,
        args.limit,
    )
    print(
        f"[sft] tokenized {len(examples)} usable examples "
        f"(dropped {len(rows)-len(examples)}: {dropped})",
        flush=True,
    )
    if len(examples) < 10:
        print("[sft] FATAL: too few usable examples to train", flush=True)
        sys.exit(1)

    if args.valid:
        valid_rows = read_jsonl(args.valid)
        valid_ex, _ = load_or_build_examples(
            valid_rows,
            tok,
            args.max_len,
            cache_dir,
            args.model,
            args.valid,
            0,
        )
        train_ex = examples
    else:
        n_val = max(1, int(len(examples) * args.valid_frac))
        valid_ex = examples[:n_val]
        train_ex = examples[n_val:]
    print(
        f"[sft] train={len(train_ex)} valid={len(valid_ex)} "
        f"attn={attn} bucket_batch={bucket_batch}",
        flush=True,
    )

    train_ds = Dataset.from_list(train_ex)
    valid_ds = Dataset.from_list(valid_ex)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation=attn,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(tok, padding="longest", label_pad_token_id=IGNORE)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=4,
        eval_strategy="steps",
        eval_steps=eval_steps,
        report_to=[],
        gradient_checkpointing=True,
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
    )

    trainer = BucketTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        bucket_batch=bucket_batch,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[sft] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
