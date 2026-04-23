#!/usr/bin/env python3
"""
OpenMythos-Mini: a MiniMind-style low-resource trainer for OpenMythos.

This script is intentionally simple:
- local text data or a tiny built-in smoke corpus
- byte-level tokenization with no external downloads
- auto device selection for CUDA, MPS, or CPU
- tiny model presets sized for local research loops

Examples:
    python training/train_mythos_mini.py --variant nano --steps 20
    python training/train_mythos_mini.py --train-data data/my_notes.txt --variant tiny
    python training/train_mythos_mini.py --train-data ./corpus --variant small --use-moe
    python training/train_mythos_mini.py --train-data . --include-code --preset deep_loops
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_mythos import ByteTokenizer, OpenMythos, mythos_nano, mythos_small, mythos_tiny
from training.mini_experiments import EXPERIMENT_PRESETS


VARIANTS = {
    "nano": mythos_nano,
    "tiny": mythos_tiny,
    "small": mythos_small,
}

SMOKE_CORPUS = (
    "OpenMythos is a recurrent-depth transformer research project.\n"
    "This bundled corpus is only for smoke tests and tiny local runs.\n"
    "Use --train-data with your own text files for real experiments.\n"
)

DEFAULT_TEXT_EXTENSIONS = {".txt", ".md"}
OPTIONAL_CODE_EXTENSIONS = {".py", ".toml", ".yaml", ".yml", ".json"}
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-maccheck",
    "__pycache__",
    "runs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument(
        "--include-ext",
        type=str,
        default=None,
        help="Comma-separated directory file extensions to include, e.g. '.md,.txt,.py'.",
    )
    parser.add_argument(
        "--include-code",
        action="store_true",
        help="Include common code/config files like .py, .toml, .json, and .yaml.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on how many files to read from a directory corpus.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Optional character cap for directory corpora to keep laptop runs bounded.",
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="nano")
    parser.add_argument(
        "--preset",
        choices=sorted(EXPERIMENT_PRESETS),
        default=None,
        help="Named small-scale research preset. Explicit CLI flags still override it.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a checkpoint path or 'latest' in --out-dir.",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--n-loops", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--dense-ffn-mult", type=float, default=None)
    parser.add_argument("--attn-type", choices=["gqa", "mla"], default=None)
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--use-act", action="store_true")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--sample-prompt", type=str, default="OpenMythos")
    parser.add_argument("--sample-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="checkpoints-mini")
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Optional JSONL file for eval metrics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def normalize_extensions(include_ext: str | None, include_code: bool) -> tuple[str, ...]:
    extensions = set(DEFAULT_TEXT_EXTENSIONS)
    if include_ext:
        extensions.update(
            f".{part.strip().lower().lstrip('.')}"
            for part in include_ext.split(",")
            if part.strip()
        )
    if include_code:
        extensions.update(OPTIONAL_CODE_EXTENSIONS)
    return tuple(sorted(extensions))


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def collect_corpus_files(
    source: Path,
    include_ext: tuple[str, ...],
    max_files: int | None,
) -> list[Path]:
    matches: list[Path] = []
    for root, dirnames, filenames in os.walk(source):
        dirnames[:] = sorted(
            name for name in dirnames if name not in DEFAULT_EXCLUDED_DIRS
        )
        root_path = Path(root)
        for filename in sorted(filenames):
            path = root_path / filename
            if path.suffix.lower() not in include_ext:
                continue
            matches.append(path)
            if max_files is not None and len(matches) >= max_files:
                return matches
    return matches


def load_corpus(
    path: str | None,
    include_ext: str | None = None,
    include_code: bool = False,
    max_files: int | None = None,
    max_chars: int | None = None,
) -> tuple[str, dict]:
    if path is None:
        corpus = SMOKE_CORPUS * 512
        return corpus, {
            "source": "smoke_corpus",
            "files_loaded": 1,
            "chars_loaded": len(corpus),
            "extensions": [],
            "truncated": False,
        }

    source = Path(path)
    if source.is_file():
        full_text = load_text_file(source)
        corpus = full_text
        if max_chars is not None:
            corpus = corpus[:max_chars]
        return corpus, {
            "source": str(source),
            "files_loaded": 1,
            "chars_loaded": len(corpus),
            "extensions": [source.suffix.lower()],
            "truncated": max_chars is not None and len(corpus) < len(full_text),
        }

    if source.is_dir():
        extensions = normalize_extensions(include_ext, include_code)
        files = collect_corpus_files(source, extensions, max_files=max_files)
        if files:
            parts = []
            chars_loaded = 0
            truncated = False
            for file in files:
                full_text = load_text_file(file)
                text = full_text
                if max_chars is not None:
                    remaining = max_chars - chars_loaded
                    if remaining <= 0:
                        truncated = True
                        break
                    text = text[:remaining]
                    if len(text) < len(full_text):
                        truncated = True
                parts.append(text)
                chars_loaded += len(text)
                if max_chars is not None and chars_loaded >= max_chars:
                    truncated = True
                    break

            corpus = "\n".join(parts)
            return corpus, {
                "source": str(source),
                "files_loaded": len(parts),
                "chars_loaded": len(corpus),
                "extensions": list(extensions),
                "truncated": truncated,
            }

    raise FileNotFoundError(
        f"Could not load text data from {path!r}. Pass a UTF-8 text file or directory."
    )


class ByteTextDataset:
    def __init__(self, text: str, tokenizer: ByteTokenizer, seq_len: int, val_fraction: float = 0.1):
        tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        if tokens.numel() == 0:
            raise ValueError("Training text is empty after loading.")
        min_tokens = max(seq_len + 2, 32)
        if tokens.numel() < min_tokens:
            repeats = (min_tokens // max(1, tokens.numel())) + 1
            tokens = tokens.repeat(repeats)

        split_idx = int(tokens.numel() * (1.0 - val_fraction))
        split_idx = min(max(split_idx, seq_len + 1), tokens.numel() - (seq_len + 1))

        self.seq_len = seq_len
        self.train_tokens = tokens[:split_idx].contiguous()
        self.val_tokens = tokens[split_idx:].contiguous()
        if self.val_tokens.numel() < seq_len + 1:
            self.val_tokens = self.train_tokens.clone()

    def get_batch(self, split: str, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_tokens if split == "train" else self.val_tokens
        max_start = data.size(0) - self.seq_len - 1
        starts = torch.randint(0, max_start + 1, (batch_size,))
        x = torch.stack([data[i : i + self.seq_len] for i in starts])
        y = torch.stack([data[i + 1 : i + self.seq_len + 1] for i in starts])
        return x.to(device), y.to(device)


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset is None:
        return args

    merged = vars(args).copy()
    sentinel_defaults = {
        "variant": "nano",
        "n_loops": None,
        "dropout": None,
        "dense_ffn_mult": None,
        "attn_type": None,
        "use_moe": False,
        "use_act": False,
    }

    for key, value in EXPERIMENT_PRESETS[args.preset].items():
        if merged.get(key) == sentinel_defaults.get(key):
            merged[key] = value

    return argparse.Namespace(**merged)


def build_config(args: argparse.Namespace, vocab_size: int):
    cfg = VARIANTS[args.variant]()
    cfg.vocab_size = vocab_size
    if args.seq_len is not None:
        cfg.max_seq_len = args.seq_len
        cfg.max_output_tokens = args.seq_len
    if args.n_loops is not None:
        cfg.max_loop_iters = args.n_loops
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.dense_ffn_mult is not None:
        cfg.dense_ffn_mult = args.dense_ffn_mult
    if args.attn_type is not None:
        cfg.attn_type = args.attn_type
    if args.use_moe:
        cfg.recurrent_use_moe = True
        cfg.n_experts = max(cfg.n_experts, 4)
        cfg.n_experts_per_tok = max(cfg.n_experts_per_tok, 1)
        cfg.expert_dim = max(cfg.expert_dim, cfg.dim)
    if args.use_act:
        cfg.use_act = True
    return cfg


def parameter_count(model: OpenMythos) -> int:
    return sum(p.numel() for p in model.parameters())


def write_metrics(metrics_file: str | None, payload: dict) -> None:
    if metrics_file is None:
        return
    path = Path(metrics_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def estimate_loss(
    model: OpenMythos,
    dataset: ByteTextDataset,
    batch_size: int,
    device: str,
    eval_iters: int,
    n_loops: int,
) -> dict[str, float]:
    losses: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for split in ("train", "val"):
            split_losses = []
            for _ in range(eval_iters):
                x, y = dataset.get_batch(split, batch_size, device)
                logits = model(x, n_loops=n_loops)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                split_losses.append(loss.item())
            losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


def generate_sample(
    model: OpenMythos,
    tokenizer: ByteTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    n_loops: int,
    temperature: float,
    top_k: int,
) -> str:
    encoded = tokenizer.encode(prompt)
    if not encoded:
        encoded = tokenizer.encode(" ")
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            n_loops=n_loops,
            temperature=temperature,
            top_k=top_k,
        )
    return tokenizer.decode(out[0].tolist())


def save_checkpoint(
    out_dir: str,
    step: int,
    model: OpenMythos,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    cfg,
    args: argparse.Namespace,
) -> Path:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "cfg": cfg,
            "args": vars(args),
        },
        path,
    )
    return path


def list_checkpoints(out_dir: str | Path) -> list[Path]:
    output_dir = Path(out_dir)
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("step_*.pt"))


def resolve_resume_path(resume: str | None, out_dir: str | Path) -> Path | None:
    if resume is None:
        return None
    if resume == "latest":
        checkpoints = list_checkpoints(out_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {out_dir!s}")
        return checkpoints[-1]
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume}")
    return path


def load_checkpoint(
    path: str | Path,
    model: OpenMythos,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str = "cpu",
) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def main() -> None:
    args = apply_preset(parse_args())
    torch.manual_seed(args.seed)

    device = detect_device(args.device)
    tokenizer = ByteTokenizer()
    resume_path = resolve_resume_path(args.resume, args.out_dir)
    if resume_path is None:
        cfg = build_config(args, tokenizer.vocab_size)
    else:
        cfg = torch.load(resume_path, map_location="cpu", weights_only=False)["cfg"]
    corpus, corpus_meta = load_corpus(
        args.train_data,
        include_ext=args.include_ext,
        include_code=args.include_code,
        max_files=args.max_files,
        max_chars=args.max_chars,
    )
    dataset = ByteTextDataset(corpus, tokenizer, cfg.max_seq_len)

    model = OpenMythos(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )
    start_step = 0
    resumed_args: dict = {}
    if resume_path is not None:
        checkpoint = load_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            scaler=scaler,
            map_location="cpu",
        )
        start_step = int(checkpoint["step"])
        resumed_args = checkpoint.get("args", {})

    def amp_context():
        return (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else nullcontext()
        )

    variant_label = resumed_args.get("variant", args.variant)
    print(f"device={device} variant={variant_label} params={parameter_count(model):,}")
    if args.preset is not None:
        print(f"preset={args.preset}")
    print(
        f"seq_len={cfg.max_seq_len} loops={cfg.max_loop_iters} "
        f"attn={cfg.attn_type} recurrent_moe={cfg.recurrent_use_moe} use_act={cfg.use_act}"
    )
    if resume_path is not None:
        print(f"resumed_from={resume_path} start_step={start_step}")
        previous_train_data = resumed_args.get("train_data")
        if previous_train_data != args.train_data:
            print(
                "note: current --train-data differs from the checkpoint's saved value "
                f"({previous_train_data!r} -> {args.train_data!r})"
            )
    print(
        f"train_tokens={dataset.train_tokens.numel():,} val_tokens={dataset.val_tokens.numel():,}"
    )
    print(
        f"corpus_source={corpus_meta['source']} files={corpus_meta['files_loaded']} "
        f"chars={corpus_meta['chars_loaded']:,}"
    )
    if corpus_meta["extensions"]:
        print(f"corpus_extensions={','.join(corpus_meta['extensions'])}")
    if corpus_meta["truncated"]:
        print("note: corpus was truncated by --max-chars")
    if args.train_data is None:
        print("using built-in smoke corpus; pass --train-data for a real experiment")
    if args.steps <= start_step:
        print(f"target steps ({args.steps}) already reached by checkpoint step {start_step}")
        return

    model.train()
    t0 = time.perf_counter()
    batch_tokens = args.batch_size * args.grad_accum * cfg.max_seq_len

    for step in range(start_step + 1, args.steps + 1):
        lr = cosine_lr(step - 1, args.steps, args.warmup_steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for _ in range(args.grad_accum):
            x, y = dataset.get_batch("train", args.batch_size, device)
            with amp_context():
                logits = model(x, n_loops=cfg.max_loop_iters)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                loss = loss / args.grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(
                model,
                dataset,
                batch_size=args.batch_size,
                device=device,
                eval_iters=args.eval_iters,
                n_loops=cfg.max_loop_iters,
            )
            dt = time.perf_counter() - t0
            tokens_per_sec = (step * batch_tokens) / max(dt, 1e-9)
            print(
                f"step={step:04d} lr={lr:.2e} train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f} grad_norm={float(grad_norm):.2f} "
                f"elapsed={dt:.1f}s tok/s={tokens_per_sec:.1f}"
            )
            write_metrics(
                args.metrics_file,
                {
                    "step": step,
                    "lr": lr,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "grad_norm": float(grad_norm),
                    "elapsed_sec": dt,
                    "tokens_per_sec": tokens_per_sec,
                    "variant": variant_label,
                    "preset": args.preset,
                    "attn_type": cfg.attn_type,
                    "recurrent_use_moe": cfg.recurrent_use_moe,
                    "use_act": cfg.use_act,
                    "n_loops": cfg.max_loop_iters,
                    "corpus_source": corpus_meta["source"],
                    "corpus_files": corpus_meta["files_loaded"],
                    "corpus_chars": corpus_meta["chars_loaded"],
                },
            )

        if args.sample_every > 0 and (step % args.sample_every == 0 or step == args.steps):
            sample = generate_sample(
                model,
                tokenizer,
                prompt=args.sample_prompt,
                device=device,
                max_new_tokens=args.sample_tokens,
                n_loops=cfg.max_loop_iters,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print("sample:")
            print(sample)

        if args.checkpoint_every > 0 and (step % args.checkpoint_every == 0 or step == args.steps):
            path = save_checkpoint(
                args.out_dir, step, model, optimizer, scaler, cfg, args
            )
            print(f"checkpoint={path}")


if __name__ == "__main__":
    main()
