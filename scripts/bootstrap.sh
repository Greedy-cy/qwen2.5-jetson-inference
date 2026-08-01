#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m venv --system-site-packages "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install \
  --index-url "${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}" \
  -r "$ROOT/requirements.txt"
echo "Python environment ready: $ROOT/.venv"
