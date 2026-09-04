"""Tests for the streamed per-chunk metrics CSV."""

import csv
import time

import numpy as np
import pytest

from audio_validation.continous_validation.criteria import AudioCriteria
from audio_validation.continous_validation.metrics_writer import COLUMNS
from audio_validation.continous_validation.validator import (
    ContinuousAudioValidator,
    ValidatorConfig,
)

SAMPLE_RATE = 8000
CHUNKS = 12


class ToneRecorder:
    """Gapless recorder handing out a fixed tone, then end-of-stream."""

    def __init__(self, chunks: int, sample_rate: int = SAMPLE_RATE) -> None:
        self._left = chunks
        self._sample_rate = sample_rate
        self.stopped = False

    def start_capture(self) -> None:
        pass

    def read_capture(self, n_samples: int) -> np.ndarray:
        if self._left <= 0 or self.stopped:
            return np.empty((0, 2))
        self._left -= 1
        t = np.arange(n_samples) / self._sample_rate
        tone = 0.5 * np.sin(2 * np.pi * 100 * t)
        return np.column_stack([tone, tone])

    def stop_capture(self) -> None:
        self.stopped = True


def _run(tmp_path, **cfg_kwargs) -> tuple:
    recorder = ToneRecorder(CHUNKS)
    cfg = ValidatorConfig(
        sample_rate=SAMPLE_RATE,
        chunk_s=1,
        artifacts_dir=str(tmp_path),
        plot_metrics=False,
        **cfg_kwargs,
    )
    validator = ContinuousAudioValidator(
        recorder=recorder, criteria=AudioCriteria(), config=cfg
    )
    with validator:
        validator.start()
        result = validator.wait(timeout=30)
    assert result is not None, "validator did not finish"
    return result


def test_metrics_csv_holds_every_analysed_row(tmp_path):
    """Every chunk lands in the CSV, one row per channel, with the result path."""
    result = _run(tmp_path)

    assert result.error is None
    csv_path = tmp_path / "continuous_metrics.csv"
    assert result.metrics_csv_path == str(csv_path)
    with open(csv_path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == list(COLUMNS)
    body = rows[1:]
    assert len(body) == sum(len(metric.channels) for metric in result.metrics)
    assert len(body) == 2 * CHUNKS  # stereo
    indexes = [int(row[0]) for row in body]
    assert indexes == sorted(indexes)
    assert {row[4] for row in body} == {"0", "1"}  # both channels present


def test_metrics_csv_is_readable_before_the_run_ends(tmp_path):
    """Rows are on disk while the run is going, not only at finalisation."""
    recorder = ToneRecorder(chunks=10_000)
    cfg = ValidatorConfig(
        sample_rate=SAMPLE_RATE,
        chunk_s=1,
        artifacts_dir=str(tmp_path),
        plot_metrics=False,
        metrics_csv_flush_chunks=1,
    )
    validator = ContinuousAudioValidator(
        recorder=recorder, criteria=AudioCriteria(), config=cfg
    )
    csv_path = tmp_path / "continuous_metrics.csv"
    try:
        validator.start()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if csv_path.exists() and len(csv_path.read_text().splitlines()) > 3:
                break
            time.sleep(0.05)
        else:
            pytest.fail("no metrics rows appeared on disk while running")
    finally:
        validator.stop()


def test_metrics_csv_can_be_disabled(tmp_path):
    """``metrics_csv_filename=None`` writes nothing and reports no path."""
    result = _run(tmp_path, metrics_csv_filename=None)

    assert result.metrics_csv_path is None
    assert not list(tmp_path.glob("*.csv"))


def test_metrics_dataframe_max_rows_keeps_the_tail(tmp_path):
    """``max_rows`` renders only the last rows, so the cost stays flat."""
    result = _run(tmp_path)

    full = result.metrics_dataframe()
    tail = result.metrics_dataframe(max_rows=6)

    assert len(full) == 2 * CHUNKS
    assert len(tail) == 6
    assert tail.iloc[-1].to_dict() == full.iloc[-1].to_dict()
    assert list(tail["index"]) == list(full["index"])[-6:]
