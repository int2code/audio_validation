# audio_validation/src/audio_validation/continuous_validation/analyser.py
"""Continuous, gapless background audio capture and chunked analysis.

Two worker tasks run on a :class:`~concurrent.futures.ThreadPoolExecutor`,
coordinated by a bounded queue:

* **chunk reader** pulls contiguous fixed-size blocks from a :class:`Recorder`,
  tags them by cumulative sample count, and enqueues them;
* **chunk analyser** computes features, evaluates the live criteria, records
  lightweight metrics, and maintains the retention buffer.

See ``docs/superpowers/specs/2026-07-09-continuous-audio-analysis-design.md``.
"""

import logging
import os
import queue
import threading
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass, replace
from typing import Callable, List, Optional

from scipy.io import wavfile

from audio_validation.audio_features import AudioFeatures
from audio_validation.continous_validation.criteria import (
    AudioCriteria,
    evaluate_chunk,
)
from audio_validation.continous_validation.models import (
    ValidationResult,
    ChannelMetric,
    ChunkMetrics,
    FailureInfo,
    RawChunk,
)
from audio_validation.continous_validation.plots import plot_metrics_timeline
from audio_validation.continous_validation.recorder import Recorder
from audio_validation.continous_validation.retention import RetentionBuffer

logger = logging.getLogger(__name__)

_QUEUE_END_MARKER = None


@dataclass(frozen=True)
class ValidatorConfig:
    """Configuration values for :class:`ContinuousAudioValidator`.

    :param sample_rate: Sample rate in Hz (used only with *capture_fn*).
    :param channels: Channel count (used only with *capture_fn*).
    :param chunk_s: Analysis chunk size in seconds.
    :param max_capture_s: Capture-time limit; ``None`` means unlimited.
    :param pre_failure_s: Seconds retained before failure; ``None`` keeps all.
    :param post_failure_s: Seconds retained/captured after a failure.
    :param stop_on_failure: Stop the run on the first armed failure.
    :param failure_consecutive: Consecutive failing chunks required to arm.
    :param artifacts_dir: Directory for persisted artifacts (for example WAV).
    :param wav_filename: Saved WAV filename.
    :param wav_dtype: Retained/saved sample dtype.
    :param queue_maxsize: Bounded queue size for producer/consumer backpressure.
    :param plot_metrics: Render a metrics-timeline plot on finalisation.
    :param plot_filename: Saved metrics-timeline plot filename.
    """

    sample_rate: int = 48000
    channels: int = 2
    chunk_s: int = 10
    max_capture_s: Optional[int] = None
    pre_failure_s: Optional[int] = 600
    post_failure_s: int = 60
    stop_on_failure: bool = True
    failure_consecutive: int = 1
    artifacts_dir: str = "test_artifacts"
    wav_filename: str = "continuous_capture.wav"
    wav_dtype: str = "float32"
    queue_maxsize: int = 4
    plot_metrics: bool = True
    plot_filename: str = "continuous_metrics.png"


class ContinuousAudioValidator:  # pylint: disable=too-many-instance-attributes
    """Capture audio gaplessly in the background and analyse it in chunks.

    Provide exactly one capture source: a streaming *recorder* (gapless) or a
    blocking *capture_fn* (wrapped in :class:`CallableRecorder`, not gapless).

    :param recorder: Streaming capture source implementing :class:`Recorder`.
    :param capture_fn: Blocking ``capture_fn(duration_s)->ndarray`` alternative.
    :param criteria: Initial :class:`AudioCriteria` (live-updatable).
    :param config: Optional :class:`ValidatorConfig` with runtime settings.
    :param play_fn: Optional callable invoked once on :meth:`start`.
    :param stop_play_fn: Optional callable invoked on stop/cleanup.
    """

    def __init__(
        self,
        *,
        recorder: Recorder,
        criteria: Optional[AudioCriteria] = None,
        config: Optional[ValidatorConfig] = None,
        play_fn: Optional[Callable[[], object]] = None,
        stop_play_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        """Build a continuous validator with recorder/capture source and config.

        Exactly one capture source is required:
        * ``recorder`` for true gapless streaming capture.

        :param recorder: Streaming capture source implementing :class:`Recorder`.
        :param criteria: Initial :class:`AudioCriteria` (live-updatable).
        :param config: Optional :class:`ValidatorConfig` with runtime settings.
        :param play_fn: Optional callable invoked once on :meth:`start`.
        :param stop_play_fn: Optional callable invoked during stop/cleanup.
        :raises ValueError: If both or neither ``recorder``/``capture_fn`` are provided.
        """
        self._cfg = config or ValidatorConfig()

        self._recorder = recorder
        self._sample_rate = self._cfg.sample_rate
        self._chunk_samples = self._cfg.chunk_s * self._sample_rate
        self._play_fn = play_fn
        self._stop_play_fn = stop_play_fn

        self._criteria = criteria or AudioCriteria()
        self._retention = RetentionBuffer(
            self._cfg.pre_failure_s, self._cfg.post_failure_s, self._sample_rate
        )

        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._abort = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=self._cfg.queue_maxsize)

        self._metrics: List[ChunkMetrics] = []
        self._failure_time_s: Optional[float] = None
        self._failure_info: Optional[FailureInfo] = None
        self._error: Optional[str] = None

        self._captured_time_s = 0.0
        self._consec_fail = 0
        self._stop_requested = False
        self._play_stopped = False
        self._recorder_stop_lock = threading.Lock()
        self._result: Optional[ValidationResult] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: List[Future] = []

    # -- public API --------------------------------------------------------
    def start(self) -> None:
        """Start playback (if any) and submit the reader/analysis tasks."""
        if self._play_fn is not None:
            self._play_fn()
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="audio-validator"
        )
        self._futures = [
            self._executor.submit(self._read_chunks),
            self._executor.submit(self._analyse_chunks),
        ]

    def update_criteria(self, criteria: AudioCriteria) -> None:
        """Atomically swap the active criteria (thread-safe)."""
        with self._state_lock:
            self._criteria = criteria

    def snapshot_metrics(self) -> List[ChunkMetrics]:
        """Return a copy of the metrics timeline collected so far."""
        with self._state_lock:
            return list(self._metrics)

    @property
    def is_running(self) -> bool:
        """Whether any worker task is still running."""
        return any(not future.done() for future in self._futures)

    def wait(self, timeout: Optional[float] = None) -> Optional[ValidationResult]:
        """Block until the run finishes, then assemble and return the result.

        Returns ``None`` if *timeout* elapses while tasks are still running.
        """
        futures_wait(self._futures, timeout=timeout)
        if self.is_running:
            return None
        return self._finish()

    def stop(self) -> ValidationResult:
        """Request stop, join tasks, run cleanup, and return the result."""
        self._stop_requested = True
        self._stop.set()
        self._stop_recorder()
        futures_wait(self._futures, timeout=30)
        return self._finish()

    def __enter__(self) -> "ContinuousAudioValidator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.is_running:
            self.stop()
        else:
            self._finish()

    # Continous audio chunks handling and analysis ------------------------------------------------

    def _enqueue_chunk(self, chunk) -> None:
        """Enqueue *item*, honouring stop/abort; never drops the sentinel."""
        while not self._abort.is_set():
            try:
                self._queue.put(chunk, timeout=0.1)
                return
            except queue.Full:
                if chunk is not _QUEUE_END_MARKER and self._stop.is_set():
                    return
                continue

    def _stop_recorder(self) -> None:
        """Stop the recorder, serialised so concurrent callers don't overlap.

        Called by :meth:`stop` (to unblock a producer parked in ``recorder.read``)
        and by the chunk reader's ``finally`` (normal teardown). ``recorder.stop_capture()``
        implementations are idempotent, so calling it more than once is safe and
        ensures a recorder started late is still stopped.
        """
        with self._recorder_stop_lock:
            try:
                self._recorder.stop_capture()
            except Exception:  # pylint: disable=broad-except
                logger.exception("recorder.stop_capture() raised")

    def _read_chunks(self) -> None:
        """
        Read chunks from the recorder and put them into the queue.
        """
        chunk_idx = 0
        total_samples = 0
        try:
            self._recorder.start_capture()
            while not self._stop.is_set() and not self._abort.is_set():

                if (
                    self._cfg.max_capture_s is not None
                    and self._captured_time_s >= self._cfg.max_capture_s
                ):
                    break

                with self._state_lock:
                    failure_time = self._failure_time_s

                if (
                    failure_time is not None
                    and self._captured_time_s >= failure_time + self._cfg.post_failure_s
                ):
                    break

                chunk = self._recorder.read_capture(self._chunk_samples)
                if chunk is None or len(chunk) == 0:
                    logger.warning("Recorder returned empty chunk, stopping.")
                    break

                chunk_start_s = total_samples / self._sample_rate
                total_samples += len(chunk)
                chunk_end_s = total_samples / self._sample_rate
                self._captured_time_s = chunk_end_s
                self._enqueue_chunk(
                    RawChunk(chunk_idx, chunk_start_s, chunk_end_s, chunk)
                )
                chunk_idx += 1
        except Exception as exc:  # pylint: disable=broad-except
            with self._state_lock:
                self._error = f"{type(exc).__name__}: {exc}"
            self._stop.set()
        finally:
            self._enqueue_chunk(_QUEUE_END_MARKER)
            self._stop_recorder()

    def _analyse_chunks(self) -> None:
        """Consume queued chunks, compute features, and record metrics.

        Runs as a worker task. Drains the queue until the ``_QUEUE_END_MARKER``
        or ``_abort`` (hard shutdown). An error is recorded and ``_stop`` is set;
        ``_abort`` is set on exit so a reader parked on :meth:`_enqueue_chunk` is released.
        """
        try:
            while True:
                try:
                    chunk = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._abort.is_set():
                        break
                    continue
                if chunk is _QUEUE_END_MARKER:
                    break
                self._validate_chunk_features(chunk)
        except Exception as exc:  # pylint: disable=broad-except
            with self._state_lock:
                if self._error is None:
                    self._error = f"chunk analyser {type(exc).__name__}: {exc}"
            self._stop.set()
        finally:
            self._abort.set()

    def _validate_chunk_features(self, chunk: RawChunk) -> None:
        """Compute features for *chunk*, evaluate criteria, and arm on failure."""
        with self._state_lock:
            criteria = self._criteria

        use_fft = criteria.expected_frequencies is not None
        features = AudioFeatures.compute(
            samples=chunk.samples,
            sample_rate=self._sample_rate,
            expected_frequencies=criteria.expected_frequencies,
            tolerance=criteria.frequency_tolerance_hz if use_fft else None,
            freq_checker=criteria.freq_checker if use_fft else None,
            skip_latency=False,
            activity_threshold=criteria.silence_rms_threshold,
        )
        logger.debug(
            "Evaluating chunk %d: start=%.1fs end=%.1fs",
            chunk.index,
            chunk.start_s,
            chunk.end_s,
        )
        ok, reason = evaluate_chunk(features, criteria)

        metric = ChunkMetrics(
            index=chunk.index,
            start_s=chunk.start_s,
            end_s=chunk.end_s,
            ok=ok,
            reason=reason,
            channels=[
                ChannelMetric(
                    rms=feat.rms,
                    thd=feat.thd,
                    thd_n=feat.thd_n,
                    detected=feat.detected,
                    peak_frequencies=(
                        list(feat.peak_frequencies)
                        if feat.peak_frequencies is not None
                        else []
                    ),
                    failed_peaks=feat.failed_peaks,
                )
                for feat in features.channel_features
            ],
        )
        with self._state_lock:
            self._metrics.append(metric)

        self._retention.add(chunk)

        if ok:
            self._consec_fail = 0
            return
        if not self._cfg.stop_on_failure:
            return
        self._consec_fail += 1
        if self._consec_fail < self._cfg.failure_consecutive:
            return
        first = False
        with self._state_lock:
            if self._failure_time_s is None:
                self._failure_time_s = chunk.end_s
                self._failure_info = FailureInfo(
                    chunk_index=chunk.index,
                    time_s=chunk.end_s,
                    reason=reason,
                    wav_offset_s=0.0,  # will be updated when wav is written
                )
                first = True
        if first:
            self._retention.freeze(chunk.end_s)

    # Finalising the run and assembling the result ------------------------------------------------

    def _drain_task_errors(self) -> None:
        """Surface any exception a worker task raised outside its own handler."""
        for future in self._futures:
            if not future.done():
                continue
            try:
                exc = future.exception()
            except CancelledError:
                continue
            if exc is not None:
                with self._state_lock:
                    if self._error is None:
                        self._error = f"{type(exc).__name__}: {exc}"

    def _finish(self) -> ValidationResult:
        """Run stop-playback cleanup, shut down the executor, assemble result."""
        if not self._play_stopped and self._stop_play_fn is not None:
            try:
                self._stop_play_fn()
            except Exception:  # pylint: disable=broad-except
                logger.exception("stop_play_fn() raised")
        self._play_stopped = True
        if not self.is_running:
            self._drain_task_errors()
            if self._executor is not None:
                self._executor.shutdown(wait=False)
        return self._assemble_result()

    def _assemble_result(self) -> ValidationResult:
        """Build the :class:`ValidationResult` (persisting the WAV once, cached)."""
        if self.is_running:
            with self._state_lock:
                metrics = list(self._metrics)
                error = self._error
            return ValidationResult(
                stopped_reason="stopped",
                failure=None,
                metrics=metrics,
                wav_path=None,
                wav_start_s=None,
                wav_end_s=None,
                total_captured_s=self._captured_time_s,
                error=error or "stop timed out: worker tasks still running",
            )

        if self._result is not None:
            return self._result

        samples = self._retention.concatenate()
        wav_path: Optional[str] = None
        wav_start = self._retention.start_s()
        wav_end = self._retention.end_s()
        if samples is not None and len(samples):
            os.makedirs(self._cfg.artifacts_dir, exist_ok=True)
            wav_path = os.path.join(self._cfg.artifacts_dir, self._cfg.wav_filename)
            wavfile.write(
                wav_path, self._sample_rate, samples.astype(self._cfg.wav_dtype)
            )
        else:
            wav_start = None
            wav_end = None

        with self._state_lock:
            failure = self._failure_info
            error = self._error
            metrics = list(self._metrics)

        if failure is not None and wav_start is not None:
            failure = replace(failure, wav_offset_s=failure.time_s - wav_start)

        if error is not None:
            stopped_reason = "capture_error"
        elif failure is not None:
            stopped_reason = "failure"
        elif self._stop_requested:
            stopped_reason = "stopped"
        else:
            stopped_reason = "max_capture_reached"

        if wav_path is not None:
            logger.info(
                "WAV covers t=%.1fs-%.1fs (%.1fs)%s",
                wav_start,
                wav_end,
                wav_end - wav_start,
                (
                    f"; failure at t={failure.time_s:.1f}s = "
                    f"{failure.wav_offset_s:.1f}s into WAV."
                    if failure is not None
                    else "."
                ),
            )

        plot_path: Optional[str] = None
        if self._cfg.plot_metrics and metrics:
            try:
                os.makedirs(self._cfg.artifacts_dir, exist_ok=True)
                plot_path = os.path.join(
                    self._cfg.artifacts_dir, self._cfg.plot_filename
                )
                plot_metrics_timeline(
                    metrics,
                    plot_path,
                    failure_time_s=failure.time_s if failure is not None else None,
                )
            except Exception:  # pylint: disable=broad-except
                logger.exception("metrics timeline plot failed")
                plot_path = None

        self._result = ValidationResult(
            stopped_reason=stopped_reason,
            failure=failure,
            metrics=metrics,
            wav_path=wav_path,
            wav_start_s=wav_start,
            wav_end_s=wav_end,
            total_captured_s=self._captured_time_s,
            error=error,
            plot_path=plot_path,
        )
        return self._result
