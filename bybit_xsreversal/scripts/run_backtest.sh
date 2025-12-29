#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
bybit-xsreversal backtest --config config/config.yaml


