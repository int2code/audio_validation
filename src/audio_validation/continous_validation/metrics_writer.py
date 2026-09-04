"""Append-only CSV sink for the per-chunk metrics timeline.

Rows are written while the run is going, one per (chunk, channel), with the
same columns as :meth:`ValidationResult.metrics_dataframe`. Two reasons this
exists rather than rendering the timeline at the end:

* an open-ended run grows the timeline by ~170k rows a day at 1 s chunks, and
  materialising all of it into a DataFrame just to log its tail costs hundreds
  of MB and seconds of CPU right when the artifacts are being written;
* the file is on disk as the run goes, so a run killed after days — power loss,
  OOM, an interrupted session — still leaves its timeline behind.

Touched only by the analysis (consumer) thread, so it needs no locking.
"""

import csv
import logging
import os
from typing import Any, Optional, TextIO

from audio_validation.continous_validation.models import (
    ChunkMetrics,
    format_timestamp,
    format_wall_clock,
)

logger = logging.getLogger(__name__)

COLUMNS = (
    "index",
    "timestamp",
    "start",
    "end",
    "ch",
    "rms",
    "thd",
    "thd_n",
    "detected",
    "ok",
    "reason",
)


class MetricsCsvWriter:
    """Stream per-chunk metrics rows to a CSV file.

    :param path: Destination CSV path; its directory is created on open.
    :param flush_every: Flush to the OS after this many appended chunks. The
        default trades a bounded loss window (the last few seconds of a run
        that dies hard) against one write syscall per chunk.
    """

    def __init__(self, path: str, flush_every: int = 60) -> None:
        self._path = path
        self._flush_every = max(1, flush_every)
        self._file: Optional[TextIO] = None
        self._writer: Optional[Any] = None
        self._since_flush = 0
        self._disabled = False
        self._rows = 0

    @property
    def path(self) -> str:
        """Destination CSV path."""
        return self._path

    def _open(self) -> None:
        """Open the file and write the header row."""
        directory = os.path.dirname(self._path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # line_buffering off: flushing is driven by ``flush_every`` instead.
        self._file = open(self._path, "w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)

    def append(self, metric: ChunkMetrics) -> None:
        """Append one row per channel of *metric*, flushing periodically.

        Never raises: a metrics sink that fails must not take the run down, so a
        write error is logged once and the writer goes quiet.

        :param metric: The analysed chunk to record.
        """
        if self._disabled:
            return
        try:
            if self._writer is None:
                self._open()
                logger.info("Streaming metrics timeline to %s", self._path)
            for channel_index, channel in enumerate(metric.channels):
                self._writer.writerow(
                    (
                        metric.index,
                        format_wall_clock(metric.start_timestamp),
                        format_timestamp(metric.start_s),
                        format_timestamp(metric.end_s),
                        channel_index,
                        channel.rms,
                        channel.thd,
                        channel.thd_n,
                        channel.detected,
                        metric.ok,
                        metric.reason,
                    )
                )
            self._rows += len(metric.channels)
            self._since_flush += 1
            if self._since_flush >= self._flush_every:
                self._file.flush()
                self._since_flush = 0
        except Exception:  # pylint: disable=broad-except
            # A metrics sink that fails must not take the run down; go quiet
            # instead, and never reopen (that would truncate what was written).
            self._disabled = True
            logger.exception("metrics CSV write failed; no further rows recorded")
            self.close()

    def close(self) -> None:
        """Flush and close the file; safe to call more than once."""
        if self._file is None:
            return
        try:
            self._file.flush()
            self._file.close()
        except Exception:  # pylint: disable=broad-except
            logger.exception("closing metrics CSV failed")
        finally:
            self._file = None
            self._writer = None

    def written(self) -> bool:
        """Whether any metrics row was appended."""
        return self._rows > 0
