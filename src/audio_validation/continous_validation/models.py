"""Lightweight data models for continuous audio analysis.

None of these retain large raw-sample arrays except :class:`RawChunk`, which is
short-lived (only referenced by the bounded retention buffer).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


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
    """Per-chunk analysis result: timing, verdict and per-channel metrics."""

    index: int
    start_s: float
    end_s: float
    ok: bool
    reason: str
    channels: List[ChannelMetric] = field(default_factory=list)


@dataclass
class FailureInfo:
    """Details of the chunk that first tripped the failure criteria."""

    chunk_index: int
    time_s: float
    reason: str
    wav_offset_s: float


@dataclass
class ValidationResult:  # pylint: disable=too-many-instance-attributes
    """Final outcome of a continuous analysis run.

    :ivar stopped_reason: one of ``"failure"``, ``"max_capture_reached"``,
        ``"stopped"``, ``"capture_error"``.
    :ivar wav_start_s/wav_end_s: coverage of the saved WAV relative to capture
        start.
    """

    stopped_reason: str
    failure: Optional[FailureInfo]
    metrics: List[ChunkMetrics]
    wav_path: Optional[str]
    wav_start_s: Optional[float]
    wav_end_s: Optional[float]
    total_captured_s: float
    error: Optional[str] = None

    def metrics_dataframe(self) -> pd.DataFrame:
        """Flatten the metrics timeline to one row per (chunk, channel)."""
        rows = []
        for metric in self.metrics:
            for ch_idx, ch_mertric in enumerate(metric.channels):
                rows.append(
                    {
                        "index": metric.index,
                        "start_s": metric.start_s,
                        "end_s": metric.end_s,
                        "ok": metric.ok,
                        "reason": metric.reason,
                        "ch": ch_idx,
                        "rms": ch_mertric.rms,
                        "thd": ch_mertric.thd,
                        "thd_n": ch_mertric.thd_n,
                        "detected": ch_mertric.detected,
                    }
                )
        return pd.DataFrame(rows)
