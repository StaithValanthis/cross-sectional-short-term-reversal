#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m unittest -v \
  tests.test_rebalance_safety \
  tests.test_risk_manager \
  tests.test_live_interval_risk_exit \
  tests.test_backtester_regression \
  tests.test_scripts
