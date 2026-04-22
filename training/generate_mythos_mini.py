#!/usr/bin/env python3
"""
Generate text from an OpenMythos-Mini checkpoint.

Examples:
    python training/generate_mythos_mini.py --checkpoint checkpoints-mini/step_000200.pt
    python training/generate_mythos_mini.py --checkpoint checkpoints-mini/step_000200.pt --prompt "Hello"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_mythos import ByteTokenizer, OpenMythos
from training.train_mythos_mini import detect_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--prompt", type=str, default="OpenMythos")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--n-loops", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    tokenizer = ByteTokenizer()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    model = OpenMythos(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = tokenizer.encode(" ")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    loops = cfg.max_loop_iters if args.n_loops is None else args.n_loops
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            n_loops=loops,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    print(f"checkpoint={args.checkpoint}")
    print(f"device={device} loops={loops}")
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
