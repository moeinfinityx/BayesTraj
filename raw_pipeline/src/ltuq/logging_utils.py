from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    level: str = "INFO",
    log_path: str | None = None,
    *,
    console_handler: logging.Handler | None = None,
) -> None:
    resolved_level = getattr(logging, level.upper(), None)
    if not isinstance(resolved_level, int):
        raise ValueError(f"Unsupported log level: {level}")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = [console_handler or logging.StreamHandler()]
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    root_logger.setLevel(resolved_level)
    for handler in handlers:
        handler.setLevel(resolved_level)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
