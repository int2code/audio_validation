"""Methods for validating captured audio against a reference WAV,
with detailed mismatch analysis and artefact generation.
"""

import logging
import os
import threading
import wave

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
save_plot_lock = threading.Lock()


def _find_mismatches(
    diff: np.ndarray,
    merge_gap: int = 16,
) -> list[tuple[int, int]]:
    """Return non-zero diff regions as ``(start, end)`` pairs.

    Regions separated by at most *merge_gap* zero samples are merged.

    :param diff: 1-D int64 difference array
    :param merge_gap: max gap in samples to merge into one run
    :return: list of ``(start, end)`` pairs (inclusive, 0-based)
    """
    mismatch_indexes = np.nonzero(diff)[0]
    if len(mismatch_indexes) == 0:
        return []
    mismatches: list[tuple[int, int]] = []
    mismatch_start = int(mismatch_indexes[0])
    last_mismatch = int(mismatch_indexes[0])
    for idx in mismatch_indexes[1:]:
        idx = int(idx)
        if idx - last_mismatch > merge_gap:
            mismatches.append((mismatch_start, last_mismatch))
            mismatch_start = idx
        last_mismatch = idx
    mismatches.append((mismatch_start, last_mismatch))
    return mismatches


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _find_best_resync_lag(
    reference: np.ndarray,
    detected: np.ndarray,
    ref_mismatch: int,
    det_mismatch: int,
    max_offset_search: int,
    resync_verify_window: int,
) -> tuple[int, float]:
    """Return the lag with the highest bit-exact match ratio at a mismatch point.

    Tries all integer lags in ``-max_offset_search … +max_offset_search``.

    :param reference: full 1-D reference array
    :param detected: full 1-D detected array
    :param ref_mismatch: absolute mismatch index in *reference*
    :param det_mismatch: absolute mismatch index in *detected*
    :param max_offset_search: maximum ±offset to try
    :param resync_verify_window: verification window length in samples
    :return: ``(best_lag, best_ratio)``
    """
    best_lag = 0
    best_ratio = 0.0
    for lag in range(-max_offset_search, max_offset_search + 1):
        ref_start = ref_mismatch + max(0, -lag)
        det_start = det_mismatch + max(0, lag)
        window = min(
            resync_verify_window,
            len(reference) - ref_start,
            len(detected) - det_start,
        )
        if window < resync_verify_window // 2:
            continue
        ratio = (
            float(
                np.sum(
                    reference[ref_start : ref_start + window].astype(np.int64)
                    == detected[det_start : det_start + window].astype(np.int64)
                )
            )
            / window
        )
        if ratio > best_ratio:
            best_ratio = ratio
            best_lag = lag
    return best_lag, best_ratio


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def _segment_compare(
    reference: np.ndarray,
    detected: np.ndarray,
    max_offset_search: int = 64,
    resync_verify_window: int = 256,
    min_match_ratio: float = 0.95,
    max_iterations: int = 128,
) -> tuple[list[tuple[int, int]], int]:
    """Compare arrays with automatic re-alignment after sample insertion/deletion glitches.

    A plain diff permanently phase-shifts after one inserted/deleted sample.
    This function re-aligns at each such event and reports only truly bad samples.

    Algorithm (up to *max_iterations*):

    1. Find first non-zero position in the remaining diff.
    2. Score candidate lags via :func:`_find_best_resync_lag`.
    3. High ratio + non-zero lag → timing glitch: record and re-align.
    4. Otherwise → data error: record run via :func:`_find_mismatches` and advance.

    :param reference: 1-D reference array (post initial-lag alignment)
    :param detected: 1-D detected array (post initial-lag alignment)
    :param max_offset_search: maximum ±offset to try at each mismatch
    :param resync_verify_window: samples used to score each candidate offset
    :param min_match_ratio: minimum match fraction to accept a re-sync
    :param max_iterations: safety cap on iterations
    :return: ``(mismatches, cumulative_lag)``
    """
    ref_pos = 0
    det_pos = 0
    mismatches: list[tuple[int, int]] = []

    for _ in range(max_iterations):
        if ref_pos >= len(reference) or det_pos >= len(detected):
            break

        remaining = min(len(reference) - ref_pos, len(detected) - det_pos)
        diff_rem = detected[det_pos : det_pos + remaining].astype(np.int64) - reference[
            ref_pos : ref_pos + remaining
        ].astype(np.int64)
        mismatch_indexes = np.nonzero(diff_rem)[0]
        if len(mismatch_indexes) == 0:
            break

        local_mismatch = int(mismatch_indexes[0])
        abs_ref_mismatch = ref_pos + local_mismatch
        abs_det_mismatch = det_pos + local_mismatch

        best_lag, best_ratio = _find_best_resync_lag(
            reference,
            detected,
            abs_ref_mismatch,
            abs_det_mismatch,
            max_offset_search,
            resync_verify_window,
        )

        if best_ratio >= min_match_ratio and best_lag != 0:
            mismatches.append((abs_ref_mismatch, abs_ref_mismatch + abs(best_lag) - 1))
            if best_lag > 0:
                # insertion in detected → advance detected
                ref_pos = abs_ref_mismatch
                det_pos = abs_det_mismatch + best_lag
            else:
                # deletion in detected → advance reference
                ref_pos = abs_ref_mismatch + abs(best_lag)
                det_pos = abs_det_mismatch
        else:
            # Data error or no clean re-sync — record run and advance past it.
            first_mismatch = _find_mismatches(diff_rem[local_mismatch:], merge_gap=16)
            if first_mismatch:
                _, mismatch_end_local = first_mismatch[0]
                mismatches.append(
                    (
                        ref_pos + local_mismatch,
                        ref_pos + local_mismatch + mismatch_end_local,
                    )
                )
                advance = local_mismatch + mismatch_end_local + 1
                ref_pos += advance
                det_pos += advance
            else:
                # Fallback: report all remaining
                mismatches.append((abs_ref_mismatch, ref_pos + remaining - 1))
                break

    return mismatches, det_pos - ref_pos  # cumulative timing drift


def _align_channel_samples(
    ref_samples: np.ndarray,
    det_samples: np.ndarray,
    channel: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Trim *det_samples* so both arrays share the same signal onset.

    :param ref_samples: 1-D reference channel samples
    :param det_samples: 1-D detected channel samples
    :param channel: channel index (error messages only)
    :return: ``(ref_aligned, det_aligned, signal_offset)``
    :raises ValueError: if either channel is completely silent
    :raises RuntimeError: if detected starts before reference (negative latency)
    """
    ref_nz = np.nonzero(ref_samples)[0]
    det_nz = np.nonzero(det_samples)[0]

    if len(ref_nz) == 0 or len(det_nz) == 0:
        raise ValueError(f"Channel {channel} is completely silent.")

    ref_start = int(ref_nz[0])
    det_start = int(det_nz[0])
    signal_offset = det_start - ref_start

    if signal_offset < 0:
        raise RuntimeError(
            f"[Ch {channel}] Detected signal starts {abs(signal_offset)} samples "
            f"BEFORE the reference (ref_start={ref_start}, det_start={det_start}). "
            f"Negative latency is physically impossible — check that detected_data "
        )

    if signal_offset > 0:
        det_aligned = det_samples[signal_offset:]
        ref_aligned = ref_samples[: len(det_aligned)]
    else:
        det_aligned = det_samples
        ref_aligned = ref_samples

    min_len = min(len(det_aligned), len(ref_aligned))
    return ref_aligned[:min_len], det_aligned[:min_len], signal_offset


def _compute_mismatch_stats(
    ref_aligned: np.ndarray,
    det_aligned: np.ndarray,
    mismatches: list[tuple[int, int]],
) -> tuple[int, int, int, str]:
    """Return ``(total_samples, max_abs_diff, n_runs, run_description)`` for mismatch runs.

    :param ref_aligned: 1-D aligned reference samples
    :param det_aligned: 1-D aligned detected samples
    :param mismatches: list of ``(start, end)`` pairs
    :return: ``(total_mismatch_samples, max_abs_diff, n_runs, run_description)``
    """
    n_runs = len(mismatches)
    total_mismatch_samples = sum(end - start + 1 for start, end in mismatches)
    max_abs_diff = max(
        int(
            np.max(
                np.abs(
                    det_aligned[s : e + 1].astype(np.int64)
                    - ref_aligned[s : e + 1].astype(np.int64)
                )
            )
        )
        for s, e in mismatches
    )
    run_description = (
        "1 continuous mismatch region"
        if n_runs == 1
        else f"{n_runs} separate mismatch regions"
    )
    return total_mismatch_samples, max_abs_diff, n_runs, run_description


def _save_mismatch_plot(
    ref_aligned: np.ndarray,
    det_aligned: np.ndarray,
    start: int,
    end: int,
    run_idx: int,
    total_runs: int,
    channel: int,
    artifacts_dir: str,
    max_plot_samples: int = 4096,
    context_samples: int = 256,
) -> str:
    """Save a PNG with reference (top) and detected (bottom) subplots for one mismatch run.

    The x-axis is capped at *max_plot_samples* for readability.

    :param ref_aligned: 1-D reference array (post-alignment)
    :param det_aligned: 1-D detected array (post-alignment)
    :param start: mismatch start index (inclusive)
    :param end: mismatch end index (inclusive)
    :param run_idx: 1-based run index
    :param total_runs: total mismatch run count
    :param channel: channel index
    :param artifacts_dir: output directory
    :param max_plot_samples: max samples to render
    :param context_samples: samples prepended/appended around the mismatch
    :return: path of the saved PNG
    """
    ctx_start = max(0, start - context_samples)
    ctx_end = min(len(ref_aligned), end + context_samples + 1)
    if ctx_end - ctx_start > max_plot_samples:
        ctx_end = ctx_start + max_plot_samples

    x = np.arange(ctx_start, ctx_end)
    ref_slice = ref_aligned[ctx_start:ctx_end]
    det_slice = det_aligned[ctx_start:ctx_end]

    span_start = max(start, ctx_start)
    span_end = min(end, ctx_end - 1)
    run_len = end - start + 1
    truncated = (ctx_end - ctx_start) >= max_plot_samples

    title_suffix = f"mismatch {run_idx}/{total_runs}  |  {run_len} bad samples" + (
        "  [plot truncated to first samples]" if truncated else ""
    )

    with save_plot_lock:
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        axes[0].plot(x, ref_slice, color="steelblue", linewidth=0.8, label="reference")
        if span_end >= span_start:
            axes[0].axvspan(
                span_start, span_end, alpha=0.25, color="red", label="mismatch"
            )
        axes[0].set_title(f"Ch {channel} – Reference  ({title_suffix})")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].plot(x, det_slice, color="darkorange", linewidth=0.8, label="detected")
        if span_end >= span_start:
            axes[1].axvspan(
                span_start, span_end, alpha=0.25, color="red", label="mismatch"
            )
        axes[1].set_title(f"Ch {channel} – Detected  ({title_suffix})")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_xlabel("Sample index (post-alignment)")
        axes[1].legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        png_path = os.path.join(
            artifacts_dir, f"null_test_ch{channel}_mismatch{run_idx:02d}.png"
        )
        plt.savefig(png_path, dpi=100)
        plt.close(fig)

    logger.info("Saved mismatch plot: %s", png_path)
    return png_path


def _save_mismatch_wav(
    ref_aligned: np.ndarray,
    det_aligned: np.ndarray,
    start: int,
    end: int,
    run_idx: int,
    channel: int,
    artifacts_dir: str,
    sample_rate: int,
    dtype: str,
    context_samples: int = 256,
) -> str:
    """Save a 2-channel WAV clip for one mismatch run (reference=L, detected=R).

    :param ref_aligned: 1-D reference array (post-alignment)
    :param det_aligned: 1-D detected array (post-alignment)
    :param start: mismatch start index (inclusive)
    :param end: mismatch end index (inclusive)
    :param run_idx: 1-based run index
    :param channel: channel index
    :param artifacts_dir: output directory
    :param sample_rate: sample rate in Hz
    :param dtype: numpy dtype string (e.g. ``"int16"``)
    :param context_samples: samples prepended/appended around the mismatch
    :return: path of the saved WAV
    """
    wav_start = max(0, start - context_samples)
    wav_end = min(len(ref_aligned), end + context_samples + 1)
    ref_wav = ref_aligned[wav_start:wav_end].reshape(-1, 1)
    det_wav = det_aligned[wav_start:wav_end].reshape(-1, 1)
    stacked = np.hstack([ref_wav, det_wav])  # 2-ch: ref=L, detected=R
    wav_path = os.path.join(
        artifacts_dir, f"null_test_ch{channel}_mismatch{run_idx:02d}.wav"
    )
    with wave.open(wav_path, "wb") as wf:  # pylint: disable=no-member
        wf.setnchannels(2)  # type: ignore[attr-defined]  # pylint: disable=no-member
        wf.setframerate(sample_rate)  # type: ignore[attr-defined]  # pylint: disable=no-member
        wf.setsampwidth(np.dtype(dtype).itemsize)  # type: ignore[attr-defined]  # pylint: disable=no-member
        wf.writeframes(stacked.astype(dtype).tobytes())  # type: ignore[attr-defined]  # pylint: disable=no-member
    logger.info("Saved mismatch WAV: %s", wav_path)
    return wav_path


def _save_mismatch_artifacts(
    ref_aligned: np.ndarray,
    det_aligned: np.ndarray,
    mismatches: list[tuple[int, int]],
    channel: int,
    artifacts_dir: str,
    max_plot_samples: int = 4096,
    context_samples: int = 256,
) -> list[str]:
    """Save a PNG per mismatch run and (when >1 run) a WAV per run.

    :param ref_aligned: 1-D reference array (post-alignment)
    :param det_aligned: 1-D detected array (post-alignment)
    :param mismatches: list of ``(start, end)`` run pairs
    :param channel: channel index
    :param artifacts_dir: output directory
    :param max_plot_samples: max samples per plot
    :param context_samples: guard-band around each run
    :return: list of created file paths
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    saved: list[str] = []
    total_runs = len(mismatches)

    for run_idx, (mismatch_start, mismatch_end) in enumerate(mismatches, start=1):
        png_path = _save_mismatch_plot(
            ref_aligned=ref_aligned,
            det_aligned=det_aligned,
            start=mismatch_start,
            end=mismatch_end,
            run_idx=run_idx,
            total_runs=total_runs,
            channel=channel,
            artifacts_dir=artifacts_dir,
            max_plot_samples=max_plot_samples,
            context_samples=context_samples,
        )
        saved.append(png_path)
    return saved


def _build_null_test_table(rows: list[dict]) -> str:
    """Format per-channel null-test results as a table string for pytest assert messages.

    :param rows: per-channel result dicts produced by :func:`validate_audio_data`
    :return: newline-prefixed table string
    """
    df = pd.DataFrame(rows).set_index("ch")
    return "\n" + df.to_string(justify="left")


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def validate_audio_perfect(
    detected_data: np.ndarray,
    reference_data: np.ndarray,
    samplerate: int,
    artifacts_dir: str = "test_artifacts",
    broadcast_reference_ch0: bool = False,
) -> tuple[bool, str]:
    """Null-test every channel after aligning detected audio to the reference.

    Alignment is based on signal onset per channel. A channel passes when every
    aligned sample difference is exactly zero (bit-perfect). Timing-only drift
    (sample insertions/deletions) is resolved by :func:`_segment_compare` and
    does not cause a failure on its own.

    On failure, artefacts are written to *artifacts_dir*:

    * PNG per mismatch run (reference + detected subplots, highlighted region).
    * WAV per mismatch run (ref=L, det=R) — only when there are multiple runs.

    :param detected_data: capture array from ``record_audio``, shape ``(n_samples, channels)``
    :param reference_data: WAV array from ``play_audio``, shape ``(n_frames, channels)``
    :param samplerate: sample rate in Hz
    :param artifacts_dir: directory for failure PNGs and WAVs
    :param broadcast_reference_ch0: compare all detected channels against reference ch 0
    :return: ``(True, table)`` on pass, ``(False, table)`` on failure —
        table has one row per channel
    """
    if detected_data.ndim != 2:
        raise ValueError(
            f"detected_data must be 2-D (n_samples, channels), got shape {detected_data.shape}."
        )
    if reference_data.ndim != 2:
        raise ValueError(
            f"reference_data must be 2-D (n_frames, channels), got shape {reference_data.shape}."
        )

    reference_channels = reference_data.shape[1]
    detected_channels = detected_data.shape[1]

    if not broadcast_reference_ch0 and reference_channels != detected_channels:
        return (
            False,
            f"Channel count mismatch: detected has {detected_channels} ch, "
            f"reference has {reference_channels} ch. "
            f"Use broadcast_reference_ch0=True to compare all against ch 0.",
        )

    channel_rows: list[dict] = []

    for channel in range(detected_channels):
        ref_ch_idx = 0 if broadcast_reference_ch0 else channel
        ref_samples = reference_data[:, ref_ch_idx]
        det_samples = detected_data[:, channel]

        row: dict = {
            "ch": channel,
            "status": "?",
            "audio_offset": "-",
            "correct_samples": 0,
            "incorrect_samples": 0,
            "glitch_count": 0,
            "artifacts_path": "-",
        }

        try:
            ref_aligned, det_aligned, signal_offset = _align_channel_samples(
                ref_samples, det_samples, channel
            )
        except ValueError:
            row["status"] = "SILENT"
            channel_rows.append(row)
            continue

        row["audio_offset"] = f"{signal_offset / samplerate:.3f}s ({signal_offset})"

        if len(ref_aligned) == 0:
            row["status"] = "NO_OVERLAP"
            channel_rows.append(row)
            continue

        row["correct_samples"] = len(ref_aligned)

        diff = det_aligned.astype(np.int64) - ref_aligned.astype(np.int64)
        if int(np.count_nonzero(diff)) == 0:
            row["status"] = "PASS"
            channel_rows.append(row)
            logger.info(
                "Channel %d: PASSED (%d aligned samples).", channel, len(ref_aligned)
            )
            continue

        mismatches, timing_drift = _segment_compare(ref_aligned, det_aligned)

        if not mismatches:
            row["status"] = "PASS"
            channel_rows.append(row)
            logger.info(
                "Channel %d: PASSED (timing drift only, %+d samples).",
                channel,
                timing_drift,
            )
            continue

        total_mismatch_samples, _, n_runs, run_description = _compute_mismatch_stats(
            ref_aligned, det_aligned, mismatches
        )
        row.update(
            {
                "status": "FAIL",
                "correct_samples": len(ref_aligned) - total_mismatch_samples,
                "incorrect_samples": total_mismatch_samples,
                "glitch_count": n_runs,
            }
        )

        logger.warning(
            "[Ch %d] FAILED — %s. Saving artefacts to '%s'.",
            channel,
            run_description,
            artifacts_dir,
        )
        _save_mismatch_artifacts(
            ref_aligned=ref_aligned,
            det_aligned=det_aligned,
            mismatches=mismatches,
            channel=channel,
            artifacts_dir=artifacts_dir,
        )
        row["artifacts_path"] = artifacts_dir

        channel_rows.append(row)

    passed = all(r["status"] == "PASS" for r in channel_rows)
    table = _build_null_test_table(channel_rows)
    return passed, table
