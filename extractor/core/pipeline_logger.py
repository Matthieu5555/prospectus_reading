"""Structured logging for the extraction pipeline.

Provides consistent logging with:
- Timestamps
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Phase/context tracking
- Structured data logging
- File output for later analysis
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class PipelineLogger:
    """Structured logger for extraction pipeline."""

    def __init__(self, name: str = "extractor", verbose: bool = False, log_dir: str | Path | None = None):
        """Initialize the pipeline logger.

        Args:
            name: Logger name.
            verbose: If True, show DEBUG level logs.
            log_dir: Directory for log files. If None, no file logging.
        """
        self.logger = logging.getLogger(name)
        self.verbose = verbose
        self._phase: str = ""
        self._phase_start: float = 0
        self._pipeline_start: float = 0
        self._log_file: Path | None = None
        self._source_file: str = ""
        self._log_dir = Path(log_dir) if log_dir else None
        self._tick_count: int = 0
        self._tick_total: int = 0

        # Configure if not already configured
        if not self.logger.handlers:
            # Console handler (colored, concise)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ConsoleFormatter())
            console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
            self.logger.addHandler(console_handler)

        self.logger.setLevel(logging.DEBUG)

    def set_verbose(self, verbose: bool):
        """Update verbose setting."""
        self.verbose = verbose
        # Update console handler level
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    def _ts(self) -> str:
        """Get current timestamp."""
        return datetime.now().strftime("%H:%M:%S")

    def _elapsed(self) -> str:
        """Get elapsed time since phase start."""
        import time
        if self._phase_start:
            elapsed = time.time() - self._phase_start
            return f"{elapsed:.1f}s"
        return ""

    def _total_elapsed(self) -> str:
        """Get total elapsed time since pipeline start."""
        import time
        if self._pipeline_start:
            elapsed = time.time() - self._pipeline_start
            mins = int(elapsed // 60)
            secs = elapsed % 60
            if mins > 0:
                return f"{mins}m {secs:.0f}s"
            return f"{secs:.1f}s"
        return ""

    def start_pipeline(self, source_file: str):
        """Mark pipeline start and set up file logging."""
        import time
        self._pipeline_start = time.time()
        self._source_file = source_file

        # Set up file logging if log_dir configured
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(source_file).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_file = self._log_dir / f"{stem}_{timestamp}.log"

            # Add file handler (captures everything including DEBUG)
            file_handler = logging.FileHandler(self._log_file, encoding="utf-8")
            file_handler.setFormatter(FileFormatter())
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

        self.logger.info(f"[{self._ts()}] Starting pipeline: {source_file}")

    def end_pipeline(self, success: bool = True, stats: dict | None = None):
        """Mark pipeline end."""
        elapsed = self._total_elapsed()
        status = "COMPLETE" if success else "FAILED"

        self.logger.info("")  # blank line
        if stats:
            self.summary(stats)

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Pipeline {status} [{elapsed}]")
        self.logger.info(f"{'='*50}")

        if self._log_file:
            self.logger.info(f"Log: {self._log_file}")

    def start_phase(self, phase: str, total: int = 0, model: str = ""):
        """Start a new pipeline phase."""
        import time
        self._phase = phase
        self._phase_start = time.time()
        self._tick_count = 0
        self._tick_total = total

        # Build phase header
        parts = [phase.upper()]
        if total > 0:
            parts.append(f"{total} items")
        if model:
            # Shorten model name for display
            short_model = model.split("/")[-1] if "/" in model else model
            parts.append(short_model)

        header = parts[0]
        if len(parts) > 1:
            header += f" ({', '.join(parts[1:])})"

        self.logger.info("")  # blank line before phase
        self.logger.info(header)

    def end_phase(self, message: str = ""):
        """End current phase."""
        # Phase end is now handled by phase_result
        self._phase = ""

    def progress(self, current: int, total: int, item: str = ""):
        """Log progress update (DEBUG level)."""
        if total > 0:
            pct = current / total * 100
            msg = f"[{current}/{total}] ({pct:.0f}%)"
            if item:
                msg += f" {item}"
            self.debug(msg)

    def tick(self, item: str = ""):
        """Log visible progress tick (INFO level).

        Use for long-running parallel tasks where user needs reassurance.
        Call this when an item completes - counter is managed internally.
        Shows:   [3/6] pages 101-150 (12.3s)
        """
        import time
        self._tick_count += 1
        total = self._tick_total
        if total > 0:
            elapsed = time.time() - self._phase_start
            count = f"[{self._tick_count}/{total}]"
            if item:
                msg = f"  {count} {item} ({elapsed:.1f}s)"
            else:
                msg = f"  {count} ({elapsed:.1f}s)"
            self.logger.info(msg)

    def debug(self, message: str, **data):
        """Log debug message (only in verbose mode)."""
        if data:
            message = f"{message} | {_format_data(data)}"
        self.logger.debug(f"[{self._ts()}] {message}")

    def info(self, message: str, **data):
        """Log info message."""
        if data:
            message = f"{message} | {_format_data(data)}"
        self.logger.info(f"  {message}")

    def warning(self, message: str, **data):
        """Log warning message."""
        if data:
            message = f"{message} | {_format_data(data)}"
        self.logger.warning(f"[{self._ts()}] WARN: {message}")

    def error(self, message: str, exc: Exception | None = None, **data):
        """Log error message."""
        if data:
            message = f"{message} | {_format_data(data)}"
        if exc:
            message = f"{message} | {type(exc).__name__}: {exc}"
        self.logger.error(f"[{self._ts()}] ERROR: {message}")

    def milestone(self, message: str, **data):
        """Log a high-level milestone (always visible, highlighted).

        Use for key pipeline events that indicate progress or decisions.
        """
        if data:
            message = f"{message} | {_format_data(data)}"
        self.logger.info(f"  -> {message}")

    def summary(self, stats: dict):
        """Log a summary block for end-of-pipeline stats."""
        lines = ["SUMMARY"]
        for key, value in stats.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")
        self.logger.info("\n".join(lines))

    def phase_result(self, phase: str, result: str, **metrics):
        """Log phase completion with key metrics.

        Args:
            phase: Phase name (e.g., "Exploration")
            result: Brief result description
            **metrics: Key-value metrics to display
        """
        elapsed = self._elapsed()
        parts = [result]
        if metrics:
            metric_parts = [f"{k}={v}" for k, v in metrics.items()]
            parts.append(", ".join(metric_parts))
        if elapsed:
            parts.append(f"[{elapsed}]")
        self.logger.info(f"  Done: {' | '.join(parts)}")

class ConsoleFormatter(logging.Formatter):
    """Console formatter - concise, optionally colored."""

    def format(self, record: logging.LogRecord) -> str:
        # The message already includes timestamp from our methods
        return record.getMessage()


class FileFormatter(logging.Formatter):
    """File formatter - includes full details for analysis."""

    def format(self, record: logging.LogRecord) -> str:
        # Include level name for file logs
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = record.levelname[:4]
        return f"{ts} [{level}] {record.getMessage()}"


def _format_data(data: dict[str, Any]) -> str:
    """Format structured data for logging."""
    parts = []
    for k, v in data.items():
        if isinstance(v, str) and len(v) > 50:
            v = v[:47] + "..."
        elif isinstance(v, list) and len(v) > 5:
            v = f"[{len(v)} items]"
        parts.append(f"{k}={v}")
    return ", ".join(parts)


# Global logger instance
_logger: PipelineLogger | None = None


def get_logger(verbose: bool = False, log_dir: str | Path | None = None) -> PipelineLogger:
    """Get or create the global pipeline logger.

    Args:
        verbose: If True, show DEBUG level logs in console.
        log_dir: Directory for log files. If provided and logger already exists,
                 updates the log directory for future file logging.
    """
    global _logger
    if _logger is None:
        _logger = PipelineLogger(verbose=verbose, log_dir=log_dir)
    else:
        if verbose and not _logger.verbose:
            _logger.set_verbose(True)
        if log_dir and not _logger._log_dir:
            _logger._log_dir = Path(log_dir)
    return _logger


def reset_logger():
    """Reset the global logger (for testing)."""
    global _logger
    if _logger:
        # Close any file handlers
        for handler in _logger.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                _logger.logger.removeHandler(handler)
    _logger = None
