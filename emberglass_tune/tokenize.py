"""Shared SFT tokenization, disk cache, and length-bucket batching."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from datasets import Dataset

IGNORE = -100


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for ln in Path(path).read_text(encoding="utf-8").split("\n"):
        ln = ln.strip()
        if ln:
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return rows


def shrink_user(msgs: list[dict], user_idx: int, fraction: float = 0.85) -> bool:
    content = msgs[user_idx]["content"]
    if len(content) < 400:
        return False
    keep = max(400, int(len(content) * fraction))
    msgs[user_idx]["content"] = content[:keep] + "\n...[truncated for length]...\n"
    return True


def tokenize_messages(msgs: list[dict], tok, max_len: int):
    """Return example dict or None if unusable."""
    msgs = [dict(m) for m in msgs]
    user_idx = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
    if user_idx is None:
        return None

    for _ in range(24):
        full_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
        full_ids = tok.encode(full_text, add_special_tokens=False)
        if len(full_ids) <= max_len:
            break
        if not shrink_user(msgs, user_idx):
            break

    prompt_text = tok.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True,
    )
    full_text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
    )
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    full_ids = tok.encode(full_text, add_special_tokens=False)
    assistant_start = len(prompt_ids)

    if len(full_ids) > max_len:
        assistant_ids = full_ids[assistant_start:]
        if not assistant_ids:
            return None
        if len(assistant_ids) >= max_len:
            assistant_ids = assistant_ids[-(max_len - 512):]
            prompt_ids = full_ids[: min(512, assistant_start)]
        else:
            room = max_len - len(assistant_ids)
            prompt_ids = full_ids[:assistant_start][-room:]
        full_ids = prompt_ids + assistant_ids
        assistant_start = len(prompt_ids)

    labels = list(full_ids)
    for i in range(min(assistant_start, len(labels))):
        labels[i] = IGNORE
    trainable = sum(1 for x in labels if x != IGNORE)
    if trainable < 32:
        return None

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def build_examples(rows: list[dict], tok, max_len: int):
    keep = []
    dropped = {"no_user": 0, "no_trainable": 0}

    for r in rows:
        msgs = r.get("messages")
        if not msgs:
            dropped["no_user"] += 1
            continue
        ex = tokenize_messages(msgs, tok, max_len)
        if ex is None:
            dropped["no_trainable"] += 1
            continue
        keep.append(ex)

    return keep, dropped


def cache_key(model: str, data_path: str | Path, max_len: int, limit: int) -> str:
    p = Path(data_path)
    stat = p.stat()
    raw = f"{model}|{max_len}|{limit}|{stat.st_mtime_ns}|{stat.st_size}|{p.resolve()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_or_build_examples(
    rows: list[dict],
    tok,
    max_len: int,
    cache_dir: str | Path | None,
    model: str,
    data_path: str | Path,
    limit: int,
):
    """Tokenize rows, optionally loading from a datasets disk cache."""
    if cache_dir:
        root = Path(cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        tag = cache_key(model, data_path, max_len, limit)
        cache_path = root / tag
        meta_path = root / f"{tag}.meta.json"
        if cache_path.exists() and meta_path.exists():
            ds = Dataset.load_from_disk(str(cache_path))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(
                f"[sft] loaded tokenized cache {cache_path} "
                f"({meta['n_examples']} examples)",
                flush=True,
            )
            return ds.to_list(), meta["dropped"]

    examples, dropped = build_examples(rows, tok, max_len)
    if cache_dir and examples:
        tag = cache_key(model, data_path, max_len, limit)
        cache_path = Path(cache_dir) / tag
        Dataset.from_list(examples).save_to_disk(str(cache_path))
        meta_path = Path(cache_dir) / f"{tag}.meta.json"
        meta_path.write_text(
            json.dumps({"n_examples": len(examples), "dropped": dropped}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[sft] saved tokenized cache -> {cache_path}", flush=True)

    return examples, dropped


class LengthGroupedBatchSampler:
    """Group similar-length sequences to cut padding waste (HF/NeMo pattern)."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        world_size: int = 1,
        seed: int = 13,
    ):
        import torch

        self.batch_size = batch_size
        self.world_size = world_size
        self.seed = seed
        self.lengths = lengths
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        import torch

        megabatch = self.batch_size * self.world_size
        indices = torch.randperm(len(self.lengths), generator=self.generator).tolist()
        indices.sort(key=lambda i: self.lengths[i])

        batches = []
        for i in range(0, len(indices), megabatch):
            chunk = indices[i : i + megabatch]
            chunk.sort(key=lambda j: self.lengths[j], reverse=True)
            for j in range(0, len(chunk), self.batch_size):
                batches.append(chunk[j : j + self.batch_size])

        order = torch.randperm(len(batches), generator=self.generator).tolist()
        for idx in order:
            yield batches[idx]

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)


def resolve_attn(name: str) -> str:
    """Pick flash_attention_2 when installed, else sdpa."""
    if name != "auto":
        return name
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"
