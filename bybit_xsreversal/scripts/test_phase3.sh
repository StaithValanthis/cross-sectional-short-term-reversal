#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m unittest -v tests.test_phase3_research
