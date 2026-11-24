"""Tests for retention of the audio around the first failure."""

import numpy as np
from scipy.io import wavfile

from audio_validation.continous_validation.criteria import AudioCriteria
from audio_validation.continous_validation.models import RawChunk
from audio_validation.continous_validation.retention import RetentionBuffer
from audio_validation.continous_validation.validator import (
    ContinuousAudioValidator,
    ValidatorConfig,
)

SAMPLE_RATE = 8000


class GlitchRecorder:
    """Gapless recorder handing out a tone, silent on the *silent_chunks* indexes."""

    def __init__(self, chunks: int, silent_chunks: set[int]) -> None:
        self._chunks = chunks
        self._silent_chunks = silent_chunks
        self._index = 0

    def start_capture(self) -> None:
        """Do nothing; the tone needs no device."""

    def read_capture(self, n_samples: int) -> np.ndarray:
        """Return the next *n_samples* frames, or an empty array at end of stream."""
        if self._index >= self._chunks:
            return np.empty((0, 2))
        t = np.arange(n_samples) / SAMPLE_RATE
        level = 0.0 if self._index in self._silent_chunks else 0.5
        self._index += 1
        tone = level * np.sin(2 * np.pi * 100 * t)
        return np.column_stack([tone, tone])

    def stop_capture(self) -> None:
        """Do nothing; the tone needs no device."""


def _chunk(index: int) -> RawChunk:
    return RawChunk(index, float(index), float(index + 1), np.zeros((SAMPLE_RATE, 2)))


def _run(tmp_path, recorder: GlitchRecorder, **cfg_kwargs):
    cfg = ValidatorConfig(
        sample_rate=SAMPLE_RATE,
        chunk_s=1,
        artifacts_dir=str(tmp_path),
        plot_metrics=False,
        metrics_csv_filename=None,
        **cfg_kwargs,
    )
    validator = ContinuousAudioValidator(
        recorder=recorder,
        criteria=AudioCriteria(require_audio_present=True),
        config=cfg,
    )
    with validator:
        validator.start()
        return validator.wait(timeout=30), validator.failure


def test_buffer_rolls_before_failure():
    """Before a failure only the last ``pre_failure_s`` seconds are kept."""
    buffer = RetentionBuffer(pre_failure_s=3, post_failure_s=2, sample_rate=SAMPLE_RATE)
    for index in range(10):
        buffer.add(_chunk(index))

    assert (buffer.start_s(), buffer.end_s()) == (7.0, 10.0)


def test_buffer_keeps_window_around_failure():
    """After freeze the pre-failure window stays and the post window is capped."""
    buffer = RetentionBuffer(pre_failure_s=3, post_failure_s=2, sample_rate=SAMPLE_RATE)
    for index in range(6):
        buffer.add(_chunk(index))
    buffer.freeze(failure_end_s=6.0)
    for index in range(6, 20):
        buffer.add(_chunk(index))

    assert (buffer.start_s(), buffer.end_s()) == (3.0, 8.0)


def test_run_to_completion_keeps_first_failure_in_wav(tmp_path):
    """With ``stop_on_failure=False`` the WAV covers the first failure, not the run end."""
    recorder = GlitchRecorder(chunks=30, silent_chunks={10, 25})
    result, validator_failure = _run(
        tmp_path, recorder, pre_failure_s=3, post_failure_s=2, stop_on_failure=False
    )

    assert result is not None, "validator did not finish"
    assert result.error is None, result.error
    assert result.stopped_reason == "max_capture_reached"
    assert len(result.metrics) == 30, "analysis must run to the end of the capture"
    assert [metric.index for metric in result.metrics if not metric.ok] == [10, 25]
    assert result.failure is not None
    assert result.failure.chunk_index == 10
    assert (result.wav_start_s, result.wav_end_s) == (8.0, 13.0)
    assert result.failure.wav_offset_s == 2.0
    assert validator_failure.chunk_index == 10
    _, samples = wavfile.read(result.wav_path)
    assert len(samples) == 5 * SAMPLE_RATE


def test_stop_on_failure_keeps_window_around_failure(tmp_path):
    """With ``stop_on_failure=True`` the run stops and the WAV covers the failure."""
    recorder = GlitchRecorder(chunks=30, silent_chunks={10})
    result, _ = _run(
        tmp_path, recorder, pre_failure_s=3, post_failure_s=2, stop_on_failure=True
    )

    assert result is not None, "validator did not finish"
    assert result.stopped_reason == "failure"
    assert result.failure is not None
    assert result.failure.chunk_index == 10
    assert (result.wav_start_s, result.wav_end_s) == (8.0, 13.0)
