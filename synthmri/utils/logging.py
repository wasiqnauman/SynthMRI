from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any


def get_logger(name: str = "synthmri", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class CSVLogger:
    """Append-only CSV metrics log; the header is fixed by the first row's keys."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="") as f:
                self._fieldnames = next(csv.reader(f))

    def log(self, row: dict[str, Any]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fieldnames).writeheader()
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore").writerow(
                {k: row.get(k, "") for k in self._fieldnames}
            )


class TensorBoardLogger:
    """Thin optional wrapper; silently no-ops if tensorboard is not installed."""

    def __init__(self, log_dir: str | Path):
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._w = SummaryWriter(str(log_dir))
        except Exception:  # pragma: no cover
            self._w = None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._w is not None:
            self._w.add_scalar(tag, value, step)

    def image(self, tag: str, image, step: int) -> None:
        if self._w is not None:
            self._w.add_image(tag, image, step)

    def close(self) -> None:
        if self._w is not None:
            self._w.close()
