#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$ROOT_DIR/bybit_xsreversal}"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "ERROR: Project directory not found: $PROJECT_DIR" >&2
  exit 1
fi

cd "$PROJECT_DIR"

if [[ -x ".venv/bin/bybit-xsreversal" ]]; then
  BIN=".venv/bin/bybit-xsreversal"
elif command -v bybit-xsreversal >/dev/null 2>&1; then
  BIN="bybit-xsreversal"
else
  echo "ERROR: bybit-xsreversal not found. Run: bash $ROOT_DIR/install.sh" >&2
  exit 1
fi

CFG="${CFG_PATH:-config/config.yaml}"
exec "$BIN" live --config "$CFG" "$@"


