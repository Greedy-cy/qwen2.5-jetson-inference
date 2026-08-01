#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$ROOT" -B "$ROOT/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-87}"
cmake --build "$ROOT/build" --parallel "$(nproc)"
ctest --test-dir "$ROOT/build" --output-on-failure
