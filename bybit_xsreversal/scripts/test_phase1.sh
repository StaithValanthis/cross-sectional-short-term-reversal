#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m unittest -v \
  bybit_xsreversal.tests.test_rebalance_safety \
  bybit_xsreversal.tests.test_risk_manager \
  bybit_xsreversal.tests.test_live_interval_risk_exit \
  bybit_xsreversal.tests.test_backtester_regression \
  bybit_xsreversal.tests.test_scripts
