"""
src/utils/logger.py
-------------------
Structured logging with console, file, and optional TensorBoard output.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


_LOGGERS: Dict[str, logging.Logger] = {}


def get_logger(name: str = "aquaagent", log_dir: Optional[str] = None,
               level: str = "INFO") -> logging.Logger:
    """
    Return (or create) a named logger with console + optional file output.

    Args:
        name:    Logger name.
        log_dir: If provided, also writes to <log_dir>/<name>.log.
        level:   Logging level string.

    Returns:
        Configured Logger instance.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


class TBLogger:
    """
    Thin wrapper around TensorBoard SummaryWriter.
    Falls back to a no-op if TensorBoard is not installed.
    """

    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled and _TB_AVAILABLE
        if self.enabled:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, global_step=step)

    def log_dict(self, metrics: Dict[str, float], step: int,
                 prefix: str = "") -> None:
        for k, v in metrics.items():
            self.log_scalar(f"{prefix}/{k}" if prefix else k, v, step)

    def close(self) -> None:
        if self.enabled and self.writer is not None:
            self.writer.close()
