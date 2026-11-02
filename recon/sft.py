"""R0.6 SFT smoke: shortlist-format training data and an MLX LoRA run.

Training prompts are rendered by the same render_prompt/SYSTEM_PROMPT the
runner uses, and the chat template is applied by the same tokenizer, so the
training format matches inference exactly.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

from .gen import mint
from .shortlist import SYSTEM_PROMPT, build_shortlist, render_prompt, target_json

TRAIN_SEED = 900_000  # disjoint from the Phase 0 eval seeds


def make_data(out_dir: Path, n_train: int, params: dict, n_valid: int = 100, seed: int = TRAIN_SEED) -> dict:
    r = random.Random(seed)
    examples = []
    k = 0
    while len(examples) < n_train + n_valid:
        size = r.choice([50, 150, 300])
        m = mint([size], 1, seed + k * 7, params)[0]
        k += 1
        lines = r.sample(m.bank, min(len(m.bank), 25))  # a slice per month keeps months diverse
        for b in lines:
            sl = build_shortlist(m, b)
            if not sl.gold_in_shortlist:
                continue  # the target would be unreachable from this prompt
            examples.append({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_prompt(sl)},
                {"role": "assistant", "content": target_json(sl)},
            ]})
    r.shuffle(examples)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, chunk in (("train", examples[:n_train]), ("valid", examples[n_train:n_train + n_valid])):
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for e in chunk:
                f.write(json.dumps(e) + "\n")
    return {"train": n_train, "valid": n_valid, "months_used": k}


def round_trip(data_dir: Path, n: int = 100) -> int:
    """Count training targets that fail to parse as answers; must be 0."""
    import re

    bad = 0
    for i, line in enumerate((data_dir / "train.jsonl").open()):
        if i >= n:
            break
        tgt = json.loads(line)["messages"][-1]["content"]
        obj = json.loads(tgt)
        bad += not all(re.fullmatch(r"C\d+", c) for c in obj["ledger"])
    return bad


def train(model: str, data_dir: Path, adapter_dir: Path, n_train: int, batch_size: int = 4,
          rank: int = 16, lr: float = 1e-4) -> None:
    """One epoch of LoRA rank 16 via mlx_lm's trainer, loss on the assistant turn only."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg = adapter_dir / "lora_config.yaml"
    cfg.write_text(
        f"model: {model}\ntrain: true\ndata: {data_dir}\nadapter_path: {adapter_dir}\n"
        f"fine_tune_type: lora\nnum_layers: -1\nbatch_size: {batch_size}\niters: {max(1, n_train // batch_size)}\n"
        f"learning_rate: {lr}\nmask_prompt: true\nmax_seq_length: 2048\ngrad_checkpoint: true\n"
        f"steps_per_report: 25\nsteps_per_eval: 250\nval_batches: 25\nsave_every: 1000\n"
        f"lora_parameters:\n  rank: {rank}\n  scale: 20.0\n  dropout: 0.0\n"
    )
    subprocess.run([sys.executable, "-m", "mlx_lm.lora", "-c", str(cfg)], check=True)
