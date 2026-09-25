"""A filesystem lease, so several processes can share one queue of units.

The imputation and prediction grids are embarrassingly parallel over units, and
the usual way to run them is to start the same command several times. Progress
files cannot arbitrate that -- they are per-process, and two workers starting
together both read the same "already done" set.

``O_CREAT | O_EXCL`` is atomic on any POSIX filesystem, so exactly one worker
wins each unit.

A stranded claim (its owner was killed before ``release`` ran) is stolen once it
is older than ``ttl_seconds``. Choose a ttl comfortably longer than the slowest
unit; too short and two workers duplicate work, too long and a crash locks the
unit out for that long.
"""
from __future__ import annotations

import os
import time
from pathlib import Path


class Claim:
    """Context manager. ``held`` is False when another worker owns the unit."""

    def __init__(self, path: Path | str, ttl_seconds: float) -> None:
        self.path = Path(path)
        self.ttl = float(ttl_seconds)
        self.held = False

    def _stale(self) -> bool:
        try:
            return (time.time() - self.path.stat().st_mtime) > self.ttl
        except FileNotFoundError:
            return False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._stale():
            self.path.unlink(missing_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as fh:
            fh.write(f"{os.getpid()}\n{time.time()}\n")
        self.held = True
        return True

    def release(self) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False

    def __enter__(self) -> "Claim":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
