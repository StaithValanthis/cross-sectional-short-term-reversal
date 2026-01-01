from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileLock:
    """
    Very small cross-platform inter-process file lock.

    - On Linux/macOS: uses fcntl.flock
    - On Windows: uses msvcrt.locking

    We only need a best-effort "single rebalance at a time" guard to prevent two
    processes (systemd + manual run) from placing duplicate orders concurrently.
    """

    path: Path
    _fh: object | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = self.path.open("a+", encoding="utf-8")
        try:
            try:
                import fcntl  # type: ignore

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except ModuleNotFoundError:
                import msvcrt  # type: ignore

                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except Exception:
            try:
                f.close()
            except Exception:
                pass
            return False
        self._fh = f
        return True

    def release(self) -> None:
        f = self._fh
        self._fh = None
        if f is None:
            return
        try:
            try:
                import fcntl  # type: ignore

                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except ModuleNotFoundError:
                import msvcrt  # type: ignore

                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            try:
                f.close()
            except Exception:
                pass


