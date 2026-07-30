"""Consistent terminal progress and diagnostic logging for visualizations."""

from __future__ import annotations

import logging
import sys
import time


def configure_logging(verbose: bool) -> logging.Logger:
    """Configure the package logger once per CLI invocation."""
    logger = logging.getLogger("cs2_visibility")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.propagate = False
    return logger


class ProgressBar:
    """Dependency-free progress bar for measurable long-running work."""

    def __init__(self, total: int, label: str, enabled: bool = True) -> None:
        self.total = max(total, 1)
        self.label = label
        self.enabled = enabled
        self.current = 0
        self._last_render = 0.0

    def update(self, current: int) -> None:
        self.current = min(current, self.total)
        if self.enabled and (self.current == self.total or time.monotonic() - self._last_render > 0.1):
            ratio = self.current / self.total
            filled = int(ratio * 24)
            sys.stderr.write(f"\r{self.label}: [{'#' * filled}{'.' * (24 - filled)}] {ratio:6.1%} ({self.current}/{self.total})")
            sys.stderr.flush()
            self._last_render = time.monotonic()

    def finish(self) -> None:
        self.update(self.total)
        if self.enabled:
            sys.stderr.write("\n")
