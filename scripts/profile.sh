#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL_DIR:-$ROOT/models/qwen2.5-0.5b-instruct}"
PRECISION="${PRECISION:-w16a16}"
LINEAR_KERNEL="${LINEAR_KERNEL:-custom}"
NEW_TOKENS="${NEW_TOKENS:-32}"
OUTPUT_NAME="${OUTPUT_NAME:-qwen_${PRECISION}}"
mkdir -p "$ROOT/profiles"
nsys profile --trace=cuda,nvtx --sample=none \
  --output "$ROOT/profiles/$OUTPUT_NAME" --force-overwrite=true \
  "$ROOT/build/llm_infer" generate --model "$MODEL" --backend cuda \
  --precision "$PRECISION" --linear-kernel "$LINEAR_KERNEL" \
  --prompt "用一句话解释大语言模型。" \
  --max-new-tokens "$NEW_TOKENS"
