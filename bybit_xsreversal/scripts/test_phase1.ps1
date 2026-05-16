Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

python -m unittest -v `
  tests.test_rebalance_safety `
  tests.test_risk_manager `
  tests.test_live_interval_risk_exit `
  tests.test_backtester_regression `
  tests.test_scripts
