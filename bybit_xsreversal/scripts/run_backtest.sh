#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
bybit-xsreversal --config config/config.yaml backtest


