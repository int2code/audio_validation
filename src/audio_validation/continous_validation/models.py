"""Lightweight data models for continuous audio analysis.

None of these retain large raw-sample arrays except :class:`RawChunk`, which is
short-lived (only referenced by the bounded retention buffer).
"""

from dataclasses import dataclass, field
from textwrap import indent
from typing import List, Optional

import numpy as np
import pandas as pd


def format_timestamp(seconds: float) -> str:
    """Render *seconds* since capture start as ``HH:MM:SS.mmm``.

    Chunk boundaries are far easier to line up with a long run's wall clock as
    ``00:15:04.023`` than as ``904.023``. Hours are not wrapped at 24.

    :param seconds: Offset from capture start in seconds.
    :return: Zero-padded ``HH:MM:SS.mmm`` string.
    """
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


@dataclass
class RawChunk:
    """One contiguous captured block, tagged with its position in the run.

    :ivar index: 0-based chunk index in capture order.
    :ivar start_s: Start time in seconds relative to capture start.
    :ivar end_s: End time in seconds relative to capture start.
    :ivar samples: 2-D array ``(n_samples, channels)``.
    """

    index: int
    start_s: float
    end_s: float
    samples: np.ndarray


@dataclass
class ChannelMetric:
    """Per-channel scalar metrics for one chunk (no raw samples)."""

    rms: float
    thd: Optional[float]
    thd_n: Optional[float]
    detected: bool
    peak_frequencies: list
    failed_peaks: Optional[list]


@dataclass
class ChunkMetrics:
    """Per-chunk analysis result: timing, verdict and per-channel metrics.

    :ivar reason: Single-line failure summary, suited to a table column.
    :ivar detail: Multiline failure report with the per-check channel tables;
        never put this in a DataFrame cell, log it separately.
    """

    index: int
    start_s: float
    end_s: float
    ok: bool
    reason: str
    detail: str = ""
    channels: List[ChannelMetric] = field(default_factory=list)


@dataclass
class FailureInfo:
    """Details of the chunk that first tripped the failure criteria."""

    chunk_index: int
    time_s: float
    reason: str
    wav_offset_s: float

    def __str__(self) -> str:
        """Return a human-readable, multiline failure summary."""
        return (
            "FailureInfo:\n"
            f"  chunk_index: {self.chunk_index}\n"
            f"  time_s: {self.time_s}\n"
            f"  wav_offset_s: {self.wav_offset_s}\n"
            "  reason:\n"
            f"{indent(self.reason, '    ')}"
        )


@dataclass
class ValidationResult:  # pylint: disable=too-many-instance-attributes
    """Final outcome of a continuous analysis run.

    :ivar stopped_reason: one of ``"failure"``, ``"max_capture_reached"``,
        ``"stopped"``, ``"capture_error"``.
    :ivar wav_start_s/wav_end_s: coverage of the saved WAV relative to capture
        start.
    :ivar plot_path: Path to the saved metrics-timeline plot, or ``None``.
    """

    stopped_reason: str
    failure: Optional[FailureInfo]
    metrics: List[ChunkMetrics]
    wav_path: Optional[str]
    wav_start_s: Optional[float]
    wav_end_s: Optional[float]
    total_captured_s: float
    error: Optional[str] = None
    plot_path: Optional[str] = None

    def metrics_dataframe(self) -> pd.DataFrame:
        """Flatten the metrics timeline to one row per (chunk, channel).

        The ``reason`` column holds only the single-line failure summary; the
        per-check channel tables live in :meth:`failure_details`, because a
        multiline table crammed into one cell wrecks the table layout.

        ``start``/``end`` are rendered as ``HH:MM:SS.mmm`` offsets from capture
        start rather than raw seconds.
        """
        rows = []
        for metric in self.metrics:
            for ch_idx, ch_mertric in enumerate(metric.channels):
                rows.append(
                    {
                        "index": metric.index,
                        "start": format_timestamp(metric.start_s),
                        "end": format_timestamp(metric.end_s),
                        "ch": ch_idx,
                        "rms": ch_mertric.rms,
                        "thd": ch_mertric.thd,
                        "thd_n": ch_mertric.thd_n,
                        "detected": ch_mertric.detected,
                        "ok": metric.ok,
                        "reason": metric.reason,
                    }
                )
        return pd.DataFrame(rows)

    def failure_details(self) -> str:
        """Render the multiline failure report of every failing chunk.

        One block per failing chunk, each with the per-check channel tables
        produced by the criteria checks. Empty when no chunk failed.
        """
        blocks = [
            f"chunk {metric.index} (t={metric.start_s:.1f}-{metric.end_s:.1f}s):\n"
            f"{indent(metric.detail or metric.reason, '  ')}"
            for metric in self.metrics
            if not metric.ok
        ]
        return "\n".join(blocks)
