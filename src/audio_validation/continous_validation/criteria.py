# audio_validation/src/audio_validation/continuous_analysis/criteria.py
"""Live-updatable pass/fail criteria for continuous audio analysis.

An :class:`AudioCriteria` instance is immutable (``frozen=True``) so it can be
swapped atomically by reference while the analysis thread reads it, without
copying.  :func:`evaluate_chunk` runs every configured check (skipping any left
at ``None``) and reports *all* failures, each check living in its own function
that logs a per-channel table.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from audio_validation.audio_features import AudioFeatures

logger = logging.getLogger(__name__)

# pylint:disable=too-many-instance-attributes


@dataclass(frozen=True)
class AudioCriteria:
    """Expected-audio parameters and pass/fail thresholds.

    When :attr:`expected_frequencies` is set, the analyser passes FFT parameters
    to :meth:`AudioFeatures.compute`, which in turn computes THD / THD+N and
    per-channel frequency detection. Checks left at ``None`` are skipped.

    :ivar expected_frequencies: Per-channel expected frequencies, e.g.
        ``[[100], [100]]``. Enables FFT/THD computation.
    :ivar frequency_tolerance_hz: Absolute FFT match tolerance in Hz.
    :ivar freq_checker: Aggregator over per-frequency matches (``all`` / ``any``).
    :ivar require_frequency: Fail if expected frequencies are not detected.
    :ivar expect_silence: Fail if any channel is non-silent.
    :ivar silence_rms_threshold: RMS at/below which a channel counts as silent;
        also used as the signal-activity threshold for feature computation.
    :ivar require_audio_present: Fail if any channel is silent (RMS <= threshold).
    :ivar expected_rms: Per-channel expected RMS values.
    :ivar rms_tolerance: Allowed absolute deviation from :attr:`expected_rms`.
    :ivar max_thd: Maximum allowed THD percent (requires expected_frequencies).
    :ivar max_thd_n: Maximum allowed THD+N percent (requires expected_frequencies).
    :ivar custom_check: Optional hook ``(features) -> (ok, reason)`` run last.
    """

    expected_frequencies: Optional[list] = None
    frequency_tolerance_hz: float = 5.0
    freq_checker: Callable = all
    require_frequency: bool = False
    expect_silence: bool = False
    silence_rms_threshold: float = 0.05
    require_audio_present: bool = False
    expected_rms: Optional[List[float]] = None
    rms_tolerance: Optional[float] = None
    max_thd: Optional[float] = None
    max_thd_n: Optional[float] = None
    custom_check: Optional[Callable[[AudioFeatures], Tuple[bool, str]]] = None


def _channel_label(ch: int) -> str:
    """Return a stable, human-readable label for channel index *ch*."""
    return f"ch{ch}"


def _format_list(freqs: Any, max_display: int = 5) -> str:
    """Format a list/array of frequencies compactly for log tables.

    Prevents long multitone arrays from breaking text wrapping.

    :param freqs: List or array of frequencies to format.
    :param max_display: Max frequencies to show before truncating.
    :return: Formatted string representation of the frequencies.
    """
    if freqs is None or (isinstance(freqs, (list, np.ndarray)) and len(freqs) == 0):
        return "[]"
    if isinstance(freqs, (float, int)):
        return f"[{freqs:.1f}]"
    freq_list = list(freqs)
    if len(freq_list) <= max_display:
        return "[" + ", ".join(f"{float(f):.1f}" for f in freq_list) + "]"
    return (
        f"[{float(freq_list[0]):.1f}, {float(freq_list[1]):.1f} "
        f"... ({len(freq_list)} total)]"
    )


def check_silence(features: AudioFeatures, threshold: float) -> Tuple[bool, str]:
    """Check that every channel is silent (RMS at/below *threshold*).

    :param features: Per-channel features to check.
    :param threshold: RMS at/below which a channel counts as silent.
    :return: ``(True, "")`` if all channels are silent, else ``(False, reason)``
        naming the non-silent channels.
    """
    rows = [
        {
            "ch": _channel_label(ch),
            "rms": feat.rms,
            "silent": feat.rms <= threshold,
        }
        for ch, feat in enumerate(features.channel_features)
    ]
    df = pd.DataFrame(rows).set_index("ch")
    logger.info("Silence check:\n%s", df.to_string())

    non_silent_df = df[~df["silent"]]
    if non_silent_df.empty:
        return True, ""
    return False, (
        f"Expected silence (RMS <= {threshold}) but audio detected on "
        f"{len(non_silent_df)} channel(s).\n{non_silent_df.to_string()}"
    )


def check_audio_present(features: AudioFeatures, threshold: float) -> Tuple[bool, str]:
    """Check that every channel carries audio (RMS strictly above *threshold*).

    Complement of :func:`check_silence`; mirrors the legacy
    ``assert_audio_present`` presence check.

    :param features: Per-channel features to check.
    :param threshold: RMS strictly above which a channel counts as present.
    :return: ``(True, "")`` if all channels are present, else ``(False, reason)``
        naming the silent channels.
    """
    rows = [
        {
            "ch": _channel_label(ch),
            "rms": feat.rms,
            "audio_present": feat.rms > threshold,
        }
        for ch, feat in enumerate(features.channel_features)
    ]
    df = pd.DataFrame(rows).set_index("ch")
    logger.info("Audio presence check:\n%s", df.to_string())

    silent_df = df[~df["audio_present"]]
    if silent_df.empty:
        return True, ""
    return False, (
        f"No audio detected (RMS <= {threshold}) on {len(silent_df)} channel(s).\n"
        f"{silent_df.to_string()}"
    )


def check_rms(
    features: AudioFeatures, expected_rms: List[float], tolerance: float
) -> Tuple[bool, str]:
    """Check that each channel's RMS is within *tolerance* of its expected value.

    :param features: Per-channel features to check.
    :param expected_rms: Per-channel expected RMS values.
    :param tolerance: Allowed absolute deviation from *expected_rms*.
    :return: ``(True, "")`` if all channels are in tolerance, else
        ``(False, reason)`` naming the out-of-tolerance channels.
    """
    rows = [
        {
            "ch": _channel_label(ch),
            "rms": feat.rms,
            "expected_rms": expected_rms[ch],
            "tolerance": tolerance,
            "rms_ok": abs(feat.rms - expected_rms[ch]) <= tolerance,
        }
        for ch, feat in enumerate(features.channel_features)
    ]
    df = pd.DataFrame(rows).set_index("ch")
    logger.info("RMS check:\n%s", df.to_string())

    failed_df = df[~df["rms_ok"]]
    if failed_df.empty:
        return True, ""
    return False, (
        f"RMS out of tolerance on {len(failed_df)} channel(s).\n{failed_df.to_string()}"
    )


def check_frequency(features: AudioFeatures) -> Tuple[bool, str]:
    """Check that the expected frequency was detected on every channel.

    :param features: Per-channel features to check (``detected`` set by the
        FFT stage of :meth:`AudioFeatures.compute`).
    :return: ``(True, "")`` if detected on all channels, else ``(False, reason)``
        naming the channels that missed the expected frequency.
    """
    rows = [
        {
            "ch": _channel_label(ch),
            "detected": feat.detected,
            "peak_frequencies_hz": (
                _format_list(feat.peak_frequencies)
                if feat.peak_frequencies is not None
                else "[]"
            ),
            "unexpected_peaks_hz": (
                _format_list(feat.failed_peaks) if feat.failed_peaks else "[]"
            ),
        }
        for ch, feat in enumerate(features.channel_features)
    ]
    df = pd.DataFrame(rows).set_index("ch")
    logger.info("Frequency presence check:\n%s", df.to_string())

    failed_df = df[~df["detected"]]
    if failed_df.empty:
        return True, ""
    return False, (
        f"Expected frequency not detected on {len(failed_df)} channel(s).\n"
        f"{failed_df.to_string()}"
    )


def check_thd(features: AudioFeatures, max_thd: float) -> Tuple[bool, str]:
    """Check that each channel's THD is at/below *max_thd* percent.

    Channels with THD not computed (``None``) are treated as passing.

    :param features: Per-channel features to check.
    :param max_thd: Maximum allowed THD as a percentage.
    :return: ``(True, "")`` if all channels are within tolerance, else
        ``(False, reason)`` naming the channels above tolerance.
    """
    rows = [
        {
            "ch": _channel_label(ch),
            "thd_percent": feat.thd,
            "tolerance": max_thd,
            "thd_ok": feat.thd is None or feat.thd <= max_thd,
        }
        for ch, feat in enumerate(features.channel_features)
    ]
    df = pd.DataFrame(rows).set_index("ch")
    logger.info("THD check:\n%s", df.to_string())

    failed_df = df[~df["thd_ok"]]
    if failed_df.empty:
        return True, ""
    return False, (
        f"THD above tolerance ({max_thd} %) on {len(failed_df)} channel(s).\n"
        f"{failed_df.to_string()}"
    )


def check_thd_n(features: AudioFeatures, max_thd_n: float) -> Tuple[bool, str]:
    """Check that each channel's THD+N is at/below *max_thd_n* percent.

    THD+N accounts for broadband noise as well as harmonics, so it catches a
    channel that is noisy but harmonically clean. Channels with THD+N not
    computed (``None``) are treated as passing.

    :param features: Per-channel features to check.
    :param max_thd_n: Maximum allowed THD+N as a percentage.
    :return: ``(True, "")`` if all channels are within tolerance, else
        ``(False, reason)`` naming the channels above tolerance.
    """
    rows = [
        {
            "ch": _channel_label(ch),
            "thd_n_percent": feat.thd_n,
            "tolerance": max_thd_n,
            "thd_n_ok": feat.thd_n is None or feat.thd_n <= max_thd_n,
        }
        for ch, feat in enumerate(features.channel_features)
    ]
    df = pd.DataFrame(rows).set_index("ch")
    logger.info("THD+N check:\n%s", df.to_string())

    failed_df = df[~df["thd_n_ok"]]
    if failed_df.empty:
        return True, ""
    return False, (
        f"THD+N above tolerance ({max_thd_n} %) on {len(failed_df)} channel(s).\n"
        f"{failed_df.to_string()}"
    )


def evaluate_chunk(
    features: AudioFeatures, criteria: AudioCriteria
) -> Tuple[bool, str]:
    """Evaluate one chunk's features against every configured check.

    Unlike a short-circuiting check, this runs *all* checks enabled by
    *criteria* and aggregates the reasons of every failing one, so a single
    chunk reports all its problems at once.

    :param features: Per-channel features from :meth:`AudioFeatures.compute`.
    :param criteria: Active criteria snapshot.
    :return: ``(True, "")`` if every configured check passes, otherwise
        ``(False, reason)`` where *reason* joins the reason of each failing
        check (one per line).
    """
    results: List[Tuple[bool, str]] = []

    if criteria.require_audio_present:
        results.append(
            check_audio_present(features, criteria.silence_rms_threshold)
        )
    if criteria.expect_silence:
        results.append(check_silence(features, criteria.silence_rms_threshold))
    if criteria.expected_rms is not None and criteria.rms_tolerance is not None:
        results.append(
            check_rms(features, criteria.expected_rms, criteria.rms_tolerance)
        )
    if criteria.require_frequency:
        results.append(check_frequency(features))
    if criteria.max_thd is not None:
        results.append(check_thd(features, criteria.max_thd))
    if criteria.max_thd_n is not None:
        results.append(check_thd_n(features, criteria.max_thd_n))
    if criteria.custom_check is not None:
        results.append(criteria.custom_check(features))

    failures = [reason for ok, reason in results if not ok]
    if failures:
        return False, "\n".join(failures)
    return True, ""
