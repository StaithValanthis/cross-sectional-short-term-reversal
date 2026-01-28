"""
Test package marker for unittest discovery.

Also ensure repo-root discovery works:
`python -m unittest discover -s bybit_xsreversal/tests -p "test_*.py" -v`

The project’s import root is `bybit_xsreversal/` (it contains the top-level `src/` package),
so add that directory to sys.path when tests are run from the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_BYBIT_XSREV_ROOT = _HERE.parents[1]  # .../bybit_xsreversal
if str(_BYBIT_XSREV_ROOT) not in sys.path:
    sys.path.insert(0, str(_BYBIT_XSREV_ROOT))

