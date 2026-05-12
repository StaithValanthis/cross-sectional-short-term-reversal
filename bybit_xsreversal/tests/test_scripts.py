from __future__ import annotations

import unittest
from pathlib import Path


class ScriptTests(unittest.TestCase):
    def test_run_live_forwards_cli_args(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_live.sh"
        content = script.read_text(encoding="utf-8")
        self.assertIn('"$@"', content)


if __name__ == "__main__":
    unittest.main()
