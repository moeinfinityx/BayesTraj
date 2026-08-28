from __future__ import annotations

import logging
import math
import sys
import time
from datetime import datetime
from typing import TextIO

from tqdm import tqdm


def _format_eta(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_finish_time(timestamp: float | None) -> str:
    if timestamp is None or not math.isfinite(timestamp):
        return "--:--:--"
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


def _truncate_value(value: str, *, limit: int = 28) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


class _ProgressLoggingHandler(logging.Handler):
    def __init__(self, display: RunProgressDisplay) -> None:
        super().__init__()
        self._display = display

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return
        self._display.write_log(message)


class RunProgressDisplay:
    def __init__(self, *, enabled: bool | None = None, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stderr
        self.enabled = self._stream.isatty() if enabled is None else enabled
        self._bar: tqdm[object] | None = None
        self._description = "run"
        self._total = 0
        self._completed = 0
        self._initial_completed = 0
        self._started_at_monotonic = time.monotonic()
        self._last_sample_id: str | None = None
        self._last_status: str | None = None
        self._last_uncertainty: float | None = None

    def create_logging_handler(self) -> logging.Handler | None:
        if not self.enabled:
            return None
        return _ProgressLoggingHandler(self)

    def start(self, *, total: int, description: str, completed: int = 0) -> None:
        self._description = description
        self._total = max(total, 0)
        self._completed = min(max(completed, 0), self._total)
        self._initial_completed = self._completed
        self._started_at_monotonic = time.monotonic()
        if not self.enabled:
            return
        self._bar = tqdm(
            total=self._total,
            initial=self._completed,
            desc=self._description,
            dynamic_ncols=True,
            file=self._stream,
            leave=True,
            unit="sample",
        )
        self._refresh()

    def advance(
        self,
        *,
        position: int,
        sample_id: str | None = None,
        status: str | None = None,
        uncertainty: object | None = None,
    ) -> None:
        target_completed = max(position, self._completed)
        if self._total:
            target_completed = min(target_completed, self._total)
        delta = target_completed - self._completed
        self._completed = target_completed
        self._last_sample_id = sample_id
        self._last_status = status
        self._last_uncertainty = float(uncertainty) if isinstance(uncertainty, (int, float)) else None
        if self._bar is not None and delta > 0:
            self._bar.update(delta)
        self._refresh()

    def write_log(self, message: str) -> None:
        if self._bar is None:
            print(message, file=self._stream)
            return
        tqdm.write(message, file=self._stream)
        self._refresh()

    def close(self, *, status: str | None = None) -> None:
        if status is not None:
            self._last_status = status
        self._refresh()
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _refresh(self) -> None:
        if self._bar is None:
            return
        self._bar.set_postfix_str(self._build_postfix(), refresh=False)
        self._bar.refresh()

    def _build_postfix(self) -> str:
        eta_seconds = self._estimate_remaining_seconds()
        finish_at = time.time() + eta_seconds if eta_seconds is not None else None
        parts = [
            f"eta={_format_eta(eta_seconds)}",
            f"finish={_format_finish_time(finish_at)}",
        ]
        if self._last_status:
            parts.append(f"status={self._last_status}")
        if self._last_sample_id:
            parts.append(f"sample={_truncate_value(self._last_sample_id)}")
        if self._last_uncertainty is not None:
            parts.append(f"uq={self._last_uncertainty:.4f}")
        return " ".join(parts)

    def _estimate_remaining_seconds(self) -> float | None:
        if self._total <= self._completed:
            return 0.0
        completed_since_start = self._completed - self._initial_completed
        if completed_since_start <= 0:
            return None
        elapsed = time.monotonic() - self._started_at_monotonic
        if elapsed <= 0:
            return None
        remaining = self._total - self._completed
        return (elapsed / completed_since_start) * remaining