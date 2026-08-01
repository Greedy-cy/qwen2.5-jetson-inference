#!/usr/bin/env python3
"""Download the official Qwen snapshot from ModelScope without importing Transformers."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from modelscope.hub.snapshot_download import snapshot_download

    path = snapshot_download(args.model_id, local_dir=str(args.output))
    print(f"model snapshot: {path}")


if __name__ == "__main__":
    main()
