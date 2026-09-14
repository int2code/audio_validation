"""Audio feature extraction and analysis utilities.

Provides :class:`ChannelFeatures` (per-channel statistics, FFT peak detection and THD
calculation) and :class:`AudioFeatures` (multi-channel container), plus helpers for
plotting, WAV export and signal-onset detection.
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import pytest
from _pytest.python_api import ApproxBase
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.io import wavfile

from audio_validation.spectrum import Spectrum

logger = logging.getLogger(__name__)
save_plot_lock = threading.Lock()

#: Ceiling applied to reported THD / THD+N percentages.  Once the residual exceeds the
#: fundamental the ratio carries no information — it only says the expected tone is not
#: there, which the frequency-detection check reports properly.  Without a ceiling a
#: missing tone yields values like 1e15 %, which stay above every threshold (so the
#: check still fails, correctly) but flatten the metrics plot and bloat the CSV.
MAX_REPORTED_DISTORTION_PCT = 1000.0

#: Harmonics counted by :attr:`ChannelFeatures.thd_h2_h5` — the 2nd through 5th, the
#: range conventionally quoted as "THD".  The field name states the range because the
#: number is part of what the figure means: quoting a THD without it is ambiguous.
DEFAULT_THD_HARMONICS = 5

#: Audio band used by :attr:`ChannelFeatures.thd_audio` and by THD+N, in Hz.
AUDIO_BAND_HZ = (20.0, 20000.0)


# pylint:disable=too-many-instance-attributes, too-many-locals, too-many-positional-arguments
# pylint:disable=too-many-arguments
@dataclass
class ChannelFeatures:
    """Holds calculated audio quantities for a single captured channel.

    This is the per-channel building block used by :class:`AudioFeatures`.
    Every field describes one audio channel extracted from a multi-channel capture.

    :cvar samples: 1-D single-channel numpy sample array.
    :cvar detected: ``True`` when all expected frequencies were found in the FFT.
    :cvar failed_peaks: List of frequency strings (Hz) that did not match any
        expected frequency; ``None`` when FFT detection was not requested.
    :cvar peak_frequencies: 1-D array of detected FFT peak frequencies in Hz;
        ``None`` when FFT detection was not requested.
    :cvar peak_amplitudes: 1-D array of normalised amplitudes at each detected
        peak; ``None`` when FFT detection was not requested.
    :cvar rms: Root-mean-square value of the channel samples.
    :cvar max: Maximum sample value.
    :cvar min: Minimum sample value.
    :cvar dbs: Level in dBFS (placeholder, populated as ``-90.0`` by default).
    :cvar mean: Arithmetic mean of the channel samples.
    :cvar thd_h2_h5: Total Harmonic Distortion over the 2nd to 5th harmonics, as a
        percentage (e.g. ``1.0`` for 1 %).  This is the range conventionally quoted as
        "THD".  Populated when FFT detection is requested; ``None`` when it is not, or
        when the expected tone is absent.
    :cvar thd_audio: Total Harmonic Distortion over *every* harmonic falling inside
        the audio band (:data:`AUDIO_BAND_HZ`), as a percentage.  A device whose
        distortion is spread over many high-order harmonics — a switching output stage,
        say — can show a low :attr:`thd_h2_h5` and a much higher :attr:`thd_audio`, so
        the two together say more than either alone.  Comparable with
        :attr:`thd_n`, which covers the same band.
    :cvar thd_n: Total Harmonic Distortion + Noise as a percentage, measured over
        20 Hz - 20 kHz with the fundamental notched out.  Unlike :attr:`thd`, it
        counts *all* non-fundamental energy (harmonics **and** broadband noise)
        relative to the fundamental, so a noisy-but-harmonically-clean signal has low
        :attr:`thd` yet high :attr:`thd_n`.  Populated when FFT detection is
        requested; ``None`` when it is not, or when the expected tone is absent.
    :cvar start_audio_offset_s: Time in seconds from the start of the capture at
        which the signal was first detected as varying; ``-1`` if the channel is
        entirely silent.
    """

    samples: np.ndarray  # 1-D single-channel array
    detected: bool = False
    failed_peaks: list = None
    peak_frequencies: list = None
    peak_amplitudes: list = None
    rms: float = 0
    max: float = 0
    min: float = 0
    dbs: float = 0
    mean: float = 0
    thd_h2_h5: float = None
    thd_audio: float = None
    thd_n: float = None
    start_audio_offset_s: int = -1

    @staticmethod
    def _compute(
        samples: np.ndarray,
        sample_rate: int = 48000,
        expected_frequencies: list = None,
        tolerance: Union[int, float] = None,
        freq_checker: Callable = None,
        start_audio_offset_s: Optional[float] = None,
        activity_threshold: float = 100,
    ) -> "ChannelFeatures":
        """Compute all audio features from a 1-D single-channel sample array.

        Basic statistics (rms, max, min, mean) are always computed.
        FFT-based frequency detection is performed only when all three of
        *expected_frequencies*, *tolerance*, and *freq_checker* are provided.

        :param samples: 1-D array of samples for a single channel.
        :param sample_rate: Sample rate of the audio in Hz.
        :param expected_frequencies: List of expected frequencies in Hz
            (e.g. ``[400, 800]``); pass ``None`` to skip FFT detection.
        :param tolerance: Absolute frequency tolerance in Hz.
        :param freq_checker: Aggregation callable — typically built-in ``all``
            or ``any``.
        :param start_audio_offset_s: Pre-computed onset offset in seconds to
            store directly, bypassing the internal
            :func:`get_audio_start_offset` call.  Pass this when the caller has
            already trimmed the samples (e.g. ``skip_latency=True`` in
            :meth:`AudioFeatures.compute`) so that the stored offset reflects
            the original capture position rather than a near-zero residual.
        :param activity_threshold: Standard-deviation threshold, in the same
            units as *samples*, passed to :func:`get_audio_start_offset` when
            *start_audio_offset_s* is not supplied.  Defaults to ``100`` (tuned
            for integer-PCM samples); use a smaller value for float-voltage
            captures.
        :return: Fully populated :class:`ChannelFeatures` instance.
            :attr:`thd_h2_h5`, :attr:`thd_audio` and :attr:`thd_n` are computed when
            FFT detection is requested; ``None`` otherwise.
        """
        samples_float = samples.astype(np.float64)
        rms_val = round(np.sqrt(np.mean(samples_float**2)), 2)
        max_val = float(np.max(samples))
        min_val = float(np.min(samples))
        mean_val = float(np.mean(samples))
        if start_audio_offset_s is None:
            start_audio_offset_s = get_audio_start_offset(
                samples, sample_rate, threshold=activity_threshold
            )

        detected = False
        failed_peaks = None
        peak_frequencies = None
        peak_amplitudes = None
        thd_h2_h5_val = None
        thd_audio_val = None
        thd_n_val = None

        if (
            expected_frequencies is not None
            and tolerance is not None
            and freq_checker is not None
        ):
            expected_freq_approx = _calculate_approx_values(
                expected_frequencies, tolerance
            )
            # One spectrum per channel, shared by peak detection, THD and THD+N so
            # every metric sees the same window, bins and scaling.
            spectrum = Spectrum(samples, sample_rate)
            peak_frequencies, peak_amplitudes = ChannelFeatures._peaks_from_spectrum(
                spectrum.freqs, spectrum.normalised_amplitudes()
            )
            fundamental_hz = _dominant_expected_frequency(
                spectrum, expected_frequencies
            )
            if fundamental_hz is not None:
                thd_h2_h5_val = ChannelFeatures.calculate_thd(spectrum, fundamental_hz)
                thd_audio_val = ChannelFeatures.calculate_thd_audio(
                    spectrum, fundamental_hz
                )
                thd_n_val = ChannelFeatures.calculate_thd_n(spectrum, fundamental_hz)
            checks = []
            failed_peaks = []

            for exp_approx in expected_freq_approx:
                checks.append(any(pf == exp_approx for pf in peak_frequencies))

            for peak_freq in peak_frequencies:
                if peak_freq not in expected_freq_approx:
                    failed_peaks.append(str(int(peak_freq)))
            detected = freq_checker(checks)

        return ChannelFeatures(
            samples=samples,
            detected=detected,
            failed_peaks=failed_peaks,
            peak_frequencies=peak_frequencies,
            peak_amplitudes=peak_amplitudes,
            rms=rms_val,
            max=max_val,
            min=min_val,
            dbs=-90.0,
            mean=mean_val,
            thd_h2_h5=thd_h2_h5_val,
            thd_audio=thd_audio_val,
            thd_n=thd_n_val,
            start_audio_offset_s=start_audio_offset_s,
        )

    @staticmethod
    def from_wav(
        filepath: str, channel: int = 0, skip_first: int = 0
    ) -> "ChannelFeatures":
        """Build :class:`ChannelFeatures` from a single channel of a WAV file.

        :param filepath: Path to the WAV file.
        :param channel: Channel to analyse (0-based index).
        :param skip_first: Number of samples to discard from the start.
        :return: :class:`ChannelFeatures` with basic statistics; ``detected``
            is ``False``.
        """
        _, data = wavfile.read(filepath)
        samples = data[:, channel] if data.ndim > 1 else data
        samples = samples[skip_first:]
        return ChannelFeatures._compute(samples=samples)

    @staticmethod
    def _full_spectrum(samples, sample_rate=48000):
        """Compute the normalised spectrum of a 1-D sample array.

        Thin wrapper over :class:`~audio_validation.spectrum.Spectrum` kept for peak
        detection and plotting, which work in relative rather than absolute terms.

        :param samples: 1-D array of samples for a single channel.
        :param sample_rate: Sample rate in Hz.
        :return: Tuple ``(frequencies, amplitudes)`` — two 1-D arrays covering every
            FFT bin, with amplitudes normalised so the maximum value is ``1.0``.
        """
        spectrum = Spectrum(samples, sample_rate)
        return spectrum.freqs, spectrum.normalised_amplitudes()

    @staticmethod
    def _peaks_from_spectrum(x_frequencies, y_amplitudes, **find_peaks_kwargs):
        """Extract peaks from a precomputed normalised spectrum.

        :param x_frequencies: 1-D array of FFT bin frequencies in Hz (all bins).
        :param y_amplitudes: 1-D array of normalised FFT amplitudes.
        :param find_peaks_kwargs: Forwarded to :func:`scipy.signal.find_peaks`;
            defaults are ``prominence=0.03, height=0.3``.
        :return: Tuple ``(frequencies, amplitudes)`` — two 1-D arrays of
            detected peak frequencies (Hz) and their normalised amplitudes.
        """
        default_kwargs = {"prominence": 0.03, "height": 0.3}
        kwargs = default_kwargs | find_peaks_kwargs
        p_idx, _ = find_peaks(y_amplitudes, **kwargs)
        return x_frequencies[p_idx], y_amplitudes[p_idx]

    @staticmethod
    def _calculate_ffts(samples, sample_rate=48000, **find_peaks_kwargs):
        """Calculate spectrum peaks from a 1-D sample array.

        :param samples: 1-D array of samples for a single channel.
        :param sample_rate: Sample rate in Hz.
        :param find_peaks_kwargs: Forwarded to :func:`scipy.signal.find_peaks`;
            defaults are ``prominence=0.03, height=0.3``.
        :return: Tuple ``(frequencies, amplitudes)`` — two 1-D arrays of
            detected peak frequencies (Hz) and their normalised amplitudes.
        """
        x_frequencies, y_amplitudes = ChannelFeatures._full_spectrum(
            samples, sample_rate
        )
        return ChannelFeatures._peaks_from_spectrum(
            x_frequencies, y_amplitudes, **find_peaks_kwargs
        )

    @staticmethod
    def _thd_from_harmonics(
        spectrum: Spectrum,
        fundamental_hz: float,
        num_harmonics: Optional[int] = None,
        upper_hz: Optional[float] = None,
    ) -> Optional[float]:
        """Root-sum-square of harmonic amplitudes over the fundamental, as a percent.

        Shared by :meth:`calculate_thd` and :meth:`calculate_thd_audio`, which differ
        only in where they stop: a harmonic count, or a frequency ceiling.

        :param spectrum: Windowed spectrum of the channel.
        :param fundamental_hz: Nominal fundamental frequency in Hz.
        :param num_harmonics: Highest harmonic to include; ``None`` for no count limit.
        :param upper_hz: Frequency ceiling in Hz; clamped to Nyquist either way.
        :return: THD as a percentage, capped at :data:`MAX_REPORTED_DISTORTION_PCT`;
            ``None`` when the fundamental cannot be found or has zero amplitude.
        """
        fund_freq, fund_amp = spectrum.peak_near(fundamental_hz)
        if fund_freq is None or fund_freq <= 0 or fund_amp <= 0:
            return None

        ceiling = (
            spectrum.nyquist if upper_hz is None else min(upper_hz, spectrum.nyquist)
        )
        harmonic_power = 0.0
        harmonic = 2
        while num_harmonics is None or harmonic <= num_harmonics:
            harmonic_hz = harmonic * fund_freq
            if harmonic_hz >= ceiling:
                break
            _, amplitude = spectrum.peak_near(harmonic_hz)
            harmonic_power += amplitude**2
            harmonic += 1

        thd = float(np.sqrt(harmonic_power) / fund_amp * 100)
        return min(thd, MAX_REPORTED_DISTORTION_PCT)

    @staticmethod
    def calculate_thd(
        spectrum: Spectrum,
        fundamental_hz: float,
        num_harmonics: int = DEFAULT_THD_HARMONICS,
    ) -> Optional[float]:
        """Calculate THD over the first *num_harmonics* harmonics.

        Uses the amplitude-ratio definition::

            THD = sqrt(A2^2 + A3^2 + ... + An^2) / A1 * 100 %

        where ``A1`` is the amplitude of the fundamental and ``A2``-``An`` those of the
        2nd through *num_harmonics*-th harmonics.  Each amplitude is the largest bin in
        a narrow search window around the nominal frequency, which tolerates the small
        offset that playback and capture clock drift always produces.  Harmonics at or
        above Nyquist are excluded.

        The fundamental is located near *fundamental_hz* rather than by taking the
        largest bin in the spectrum, so a DC offset or an unrelated spur cannot be
        mistaken for it.

        This is the range conventionally quoted as "THD".  It says nothing about
        harmonics above the *num_harmonics*-th — for a device whose distortion runs to
        high order, see :meth:`calculate_thd_audio`.

        A harmonic buried in the noise floor reads as the largest noise bin in its
        search window rather than as zero, which biases the result slightly upward when
        the harmonics are near the floor.

        :param spectrum: Windowed :class:`~audio_validation.spectrum.Spectrum` of the
            channel.
        :param fundamental_hz: Nominal fundamental frequency in Hz — normally the
            frequency the signal generator was asked to produce.
        :param num_harmonics: Highest harmonic to include (default
            :data:`DEFAULT_THD_HARMONICS`, covering harmonics 2-5).
        :return: THD as a percentage (e.g. ``1.0`` for 1 % THD), capped at
            :data:`MAX_REPORTED_DISTORTION_PCT`; ``None`` when the fundamental cannot
            be found or has zero amplitude.
        """
        return ChannelFeatures._thd_from_harmonics(
            spectrum, fundamental_hz, num_harmonics=num_harmonics
        )

    @staticmethod
    def calculate_thd_audio(
        spectrum: Spectrum,
        fundamental_hz: float,
        band_hz: tuple = AUDIO_BAND_HZ,
    ) -> Optional[float]:
        """Calculate THD over every harmonic inside the audio band.

        Same definition as :meth:`calculate_thd` but with no harmonic count: every
        integer multiple of the fundamental up to the top of *band_hz* is included.

        Distortion does not always concentrate in the low-order harmonics.  A switching
        output stage or a quantiser spreads it over a long series reaching to the top of
        the band, and counting only the first four then reports a small fraction of what
        is there.  Measuring both makes the difference visible instead of hiding it in
        the choice of harmonic count.

        Because it covers the same band as :meth:`calculate_thd_n`, the two are directly
        comparable: when they agree, the residual is essentially all harmonic; when
        THD+N is much the larger, there is real broadband noise underneath.

        :param spectrum: Windowed :class:`~audio_validation.spectrum.Spectrum` of the
            channel.
        :param fundamental_hz: Nominal fundamental frequency in Hz.
        :param band_hz: ``(low, high)`` band in Hz; only the upper edge is used, clamped
            to Nyquist.  Defaults to :data:`AUDIO_BAND_HZ`.
        :return: THD as a percentage, capped at :data:`MAX_REPORTED_DISTORTION_PCT`;
            ``None`` when the fundamental cannot be found or has zero amplitude.
        """
        return ChannelFeatures._thd_from_harmonics(
            spectrum, fundamental_hz, upper_hz=band_hz[1]
        )

    @staticmethod
    def calculate_thd_n(
        spectrum: Spectrum,
        fundamental_hz: float,
        notch_octaves: float = 0.5,
        band_hz: tuple = AUDIO_BAND_HZ,
    ) -> Optional[float]:
        """Calculate Total Harmonic Distortion + Noise (THD+N) from a spectrum.

        THD+N extends :meth:`calculate_thd`: where THD counts only the energy at
        integer harmonics of the fundamental, THD+N counts **every** component other
        than the fundamental — harmonics *and* broadband noise, hum, aliasing.  A
        signal whose distortion is purely random noise therefore has a low
        :meth:`calculate_thd` but a high THD+N::

            THD+N = RMS(everything in band, fundamental notched out) / RMS(fundamental)

        The fundamental is removed with a notch *notch_octaves* wide either side of it,
        rather than a fixed number of bins.  A fixed bin count cannot work: the width
        of the fundamental in bins depends on the window, on the buffer length, and on
        whether the tone happens to sit on a bin, so a notch narrow enough to be
        meaningful at one buffer length leaks the fundamental into the residual at
        another and reads it as noise.

        Measuring over 20 Hz - 20 kHz rather than the whole spectrum keeps the result
        comparable with instrument readings and excludes out-of-band converter noise.

        :param spectrum: Windowed :class:`~audio_validation.spectrum.Spectrum` of the
            channel.
        :param fundamental_hz: Nominal fundamental frequency in Hz.
        :param notch_octaves: Half-width of the notch around the fundamental, in
            octaves (default ``0.5``).
        :param band_hz: ``(low, high)`` measurement band in Hz, clamped to Nyquist.
            Defaults to :data:`AUDIO_BAND_HZ`, the same band as :meth:`calculate_thd_audio`.
        :return: THD+N as a percentage (e.g. ``1.0`` for 1 %), capped at
            :data:`MAX_REPORTED_DISTORTION_PCT`; ``None`` when the fundamental cannot
            be found or has zero energy.
        """
        fund_freq, _ = spectrum.peak_near(fundamental_hz)
        if fund_freq is None or fund_freq <= 0:
            return None

        band_low, band_high = band_hz
        band_high = min(band_high, spectrum.nyquist)
        notch_low = fund_freq / 2**notch_octaves
        notch_high = fund_freq * 2**notch_octaves

        fundamental_rms = spectrum.band_rms(notch_low, notch_high)
        if fundamental_rms <= 0:
            return None

        residual_rms = float(
            np.hypot(
                spectrum.band_rms(band_low, notch_low),
                spectrum.band_rms(notch_high, band_high),
            )
        )
        thd_n = residual_rms / fundamental_rms * 100
        return min(thd_n, MAX_REPORTED_DISTORTION_PCT)


@dataclass
class AudioFeatures:
    """Full multi-channel audio capture result.

    Combines the raw multi-channel sample array with a per-channel list of
    :class:`ChannelFeatures` computed from those samples.

    :cvar samples: 2-D numpy array of shape ``(n_samples, n_channels)`` — the
        native sounddevice / interleaved layout as returned by ``record_audio``.
    :cvar channel_features: One :class:`ChannelFeatures` per channel in
        channel-index order.  Each entry may have been computed from a trimmed
        slice of the corresponding column in ``samples`` (e.g. when
        *skip_first* or *skip_latency* is used in :meth:`compute`).
    """

    samples: np.ndarray  # (n_samples, n_channels)
    channel_features: list

    def __len__(self) -> int:
        """Return the number of channels."""
        return len(self.channel_features)

    def __getitem__(self, channel: int) -> ChannelFeatures:
        """Return the :class:`ChannelFeatures` for *channel* (0-based index).

        :param channel: 0-based channel index.
        :return: :class:`ChannelFeatures` for the requested channel.
        """
        return self.channel_features[channel]

    @staticmethod
    def compute(
        samples: np.ndarray,
        sample_rate: int = 48000,
        expected_frequencies: list = None,
        tolerance: Union[int, float] = None,
        freq_checker: Callable = None,
        skip_first: int = 0,
        skip_latency: bool = False,
        activity_threshold: float = 100,
    ) -> "AudioFeatures":
        """Build :class:`AudioFeatures` from a multi-channel sample array.

        Computes :class:`ChannelFeatures` for every channel in *samples*.
        FFT-based frequency detection is performed only when
        *expected_frequencies*, *tolerance*, and *freq_checker* are all given.

        :param samples: 2-D array of shape ``(n_samples, n_channels)`` — the
            native sounddevice / interleaved layout.
        :param sample_rate: Sample rate in Hz.
        :param expected_frequencies: Per-channel expected frequencies indexed by
            channel position, e.g. ``[[400], [800]]``; pass ``None`` to skip
            FFT detection.
        :param tolerance: Absolute frequency tolerance in Hz.
        :param freq_checker: Aggregation callable — typically built-in ``all``
            or ``any``.
        :param skip_first: Samples to discard from the front of each channel
            (ignored when *skip_latency* is ``True``).
        :param skip_latency: When ``True``, auto-detect the signal start and
            trim the silent prefix instead of using *skip_first*.
        :param activity_threshold: Standard-deviation threshold, in the same
            units as *samples*, used to detect signal onset (see
            :func:`get_audio_start_offset`).  Defaults to ``100`` (tuned for
            integer-PCM samples); pass a smaller value for float-voltage
            captures so that *skip_latency* and the ``start_audio_offset_s``
            field behave correctly.
        :return: :class:`AudioFeatures` with ``samples`` and
            ``channel_features`` populated.
        """
        n_channels = samples.shape[1] if samples.ndim > 1 else 1
        channel_features: list[ChannelFeatures] = []

        for ch in range(n_channels):
            ch_samples = samples[:, ch] if samples.ndim > 1 else samples

            pre_trim_offset_s: Optional[float] = None
            if skip_latency:
                pre_trim_offset_s = get_audio_start_offset(
                    ch_samples, sample_rate, threshold=activity_threshold
                )
                if pre_trim_offset_s >= 0:
                    ch_samples = ch_samples[int(pre_trim_offset_s * sample_rate) :]
                    logger.debug(
                        "skip_latency: channel %d trimmed %.3f s (%.0f samples)",
                        ch,
                        pre_trim_offset_s,
                        pre_trim_offset_s * sample_rate,
                    )
                else:
                    logger.debug(
                        "skip_latency: channel %d is entirely silent, no trimming.", ch
                    )
            elif skip_first > 0:
                ch_samples = ch_samples[skip_first:]
                logger.debug(
                    "skip_first: channel %d trimmed %d samples (%.3f s)",
                    ch,
                    skip_first,
                    skip_first / sample_rate,
                )

            ch_freqs = (
                expected_frequencies[ch] if expected_frequencies is not None else None
            )
            ch_features = ChannelFeatures._compute(  # pylint: disable=protected-access
                samples=ch_samples,
                sample_rate=sample_rate,
                expected_frequencies=ch_freqs,
                tolerance=tolerance,
                freq_checker=freq_checker,
                start_audio_offset_s=pre_trim_offset_s,
                activity_threshold=activity_threshold,
            )
            channel_features.append(ch_features)

        return AudioFeatures(samples=samples, channel_features=channel_features)

    @staticmethod
    def from_wav(
        filepath: str,
        channels: int = 1,
        sample_rate: int = 48000,
        expected_frequencies: list = None,
        tolerance: Union[int, float] = None,
        freq_checker: Callable = None,
        skip_first: int = 0,
        skip_latency: bool = False,
        activity_threshold: float = 100,
    ) -> "AudioFeatures":
        """Build :class:`AudioFeatures` from a WAV file (no FFT detection).

        :param filepath: Path to the WAV file.
        :param channels: Number of channels to load (first *channels* tracks).
        :param skip_first: Samples to discard from the front of every channel.
        :param skip_latency: If ``True``, automatically skip initial silence
            based on *activity_threshold*.
        :param activity_threshold: Standard-deviation threshold, in the same
            units as *samples*, used to detect signal onset (see
            :func:`get_audio_start_offset`).  Defaults to ``100`` (tuned for
            integer-PCM samples); pass a smaller value for float-voltage
            captures so that *skip_latency* and the ``start_audio_offset_s``
            field behave correctly.
        :return: :class:`AudioFeatures` with basic statistics per channel;
            ``detected`` is ``False`` on every :class:`ChannelFeatures`.
        """
        _, data = wavfile.read(filepath)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        wav_samples = data[:, :channels]
        if skip_first:
            wav_samples = wav_samples[skip_first:]
        return AudioFeatures.compute(
            samples=wav_samples,
            sample_rate=sample_rate,
            expected_frequencies=expected_frequencies,
            tolerance=tolerance,
            freq_checker=freq_checker,
            skip_latency=skip_latency,
            activity_threshold=activity_threshold,
        )


def _dominant_expected_frequency(
    spectrum: Spectrum, expected_frequencies: list
) -> Optional[float]:
    """Return whichever expected frequency carries the most energy.

    THD and THD+N are defined around a single fundamental.  When only one frequency is
    expected that is the fundamental; when several are (a multitone signal) the
    strongest is used, which keeps the measurement well defined even though harmonic
    distortion of one tone is not a meaningful quantity on a multitone signal — its
    harmonics land on, or near, the other tones.

    :param spectrum: Windowed spectrum of the channel.
    :param expected_frequencies: Frequencies in Hz the channel is expected to carry.
    :return: The expected frequency with the largest measured amplitude, or ``None``
        when none of them is present.
    """
    if not expected_frequencies:
        return None
    located = [
        (amplitude, frequency)
        for frequency, amplitude in (
            (frequency, spectrum.peak_near(frequency)[1])
            for frequency in expected_frequencies
        )
        if amplitude > 0
    ]
    if not located:
        return None
    return float(max(located)[1])


def _calculate_approx_values(
    frequencies: list[int], tolerance: int | float
) -> list[ApproxBase]:
    """Return pytest.approx wrappers for each frequency ± tolerance.

    :param frequencies: List of frequency values in Hz to wrap in
        :func:`pytest.approx`.
    :param tolerance: Absolute tolerance applied symmetrically to each
        frequency.
    :return: List of :class:`~_pytest.python_api.ApproxBase` objects, one per
        input frequency.
    """
    expected_freq_approx = list(
        map(lambda x: pytest.approx(x, abs=tolerance), frequencies)
    )
    return expected_freq_approx


def draw_plots(
    audio: "AudioFeatures",
    path: str,
    chunk_size: int = 1024,
) -> None:
    """Draw time-domain and FFT plots for all channels and save to *path*.

    Produces a grid of ``(n_channels, 2)`` subplots: the left column shows the
    time-domain waveform and the right column shows the detected FFT peaks.

    :param audio: :class:`AudioFeatures` containing per-channel features.
    :param path: Destination file path including extension
        (e.g. ``"output.png"``).
    :param chunk_size: Number of samples shown on the time-domain axis.
    """
    n = len(audio.channel_features)
    with save_plot_lock:
        fig, axes = plt.subplots(n, 2, figsize=(18, 5 * n), squeeze=False)
        for ch, feat in enumerate(audio.channel_features):
            n_show = min(len(feat.samples), chunk_size * 10)

            ax_time = axes[ch][0]
            ax_time.plot(feat.samples[:n_show])
            ax_time.set_title(f"Audio CH{ch}")
            ax_time.set_xlabel(f"First {n_show} samples")
            ax_time.set_ylabel("Amplitude")

            ax_fft = axes[ch][1]
            if feat.peak_frequencies is not None and len(feat.peak_frequencies):
                ax_fft.plot(feat.peak_frequencies, feat.peak_amplitudes, "x")
                ax_fft.vlines(feat.peak_frequencies, 0, feat.peak_amplitudes)
            ax_fft.set_title(f"RFFT CH{ch}")
            ax_fft.set_xlabel("Frequency [Hz]")
            ax_fft.set_ylabel("Power")
            ax_fft.ticklabel_format(useOffset=False)

        plt.tight_layout()
        if dirname := os.path.dirname(path):
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(path)
        plt.close(fig)


def save_to_wave(
    audio: "AudioFeatures",
    path: str,
    samplerate: int = 48000,
    dtype: str = "float32",
) -> None:
    """Save all channels to a single multi-channel WAV file.

    Uses ``audio.samples`` (the original unprocessed 2-D capture array) so
    that the full, untrimmed data is written regardless of any *skip_first* /
    *skip_latency* trimming applied during feature computation.

    :param audio: :class:`AudioFeatures` whose ``samples`` array is written.
    :param path: Destination WAV file path.
    :param samplerate: Sample rate in Hz.
    :param dtype: Numpy dtype string matching the original capture format.
    """
    if dirname := os.path.dirname(path):
        os.makedirs(dirname, exist_ok=True)
    # audio.samples shape: (n_samples, n_channels) — already interleaved, write directly
    wavfile.write(path, samplerate, audio.samples.astype(dtype))


def get_audio_start_offset(
    samples: np.ndarray[Any], sample_rate: int, threshold: int = 100
) -> int | float:
    """Calculate the time (in seconds) when the audio starts varying.

    :param samples: Numpy array of audio samples.
    :param sample_rate: Sample rate of the audio in Hz.
    :param threshold: Standard-deviation threshold used to detect signal
        activity.
    :return: Time in seconds (relative to the start of *samples*) at which
        the signal first becomes active (std > *threshold*); ``-1`` if the
        signal never varies.
    """
    window = 100
    signal_evaluation = detect_if_signal_changes(
        samples, window_size=window, threshold=threshold
    )
    first_index_where_started = np.where(signal_evaluation)[0]
    if first_index_where_started.size:
        idx = first_index_where_started[0] * window
        time_where_started = idx / sample_rate
    else:
        time_where_started = -1
    return time_where_started


def detect_if_signal_changes(
    samples: np.ndarray[Any], window_size=100, threshold=50
) -> np.ndarray[np.bool]:
    """Detect whether the signal is varying within successive windows.

    Divides *samples* into non-overlapping windows of *window_size* and
    computes the standard deviation of each window.  A window is marked as
    active when its standard deviation exceeds *threshold*.

    :param samples: Numpy array of audio samples.
    :param window_size: Number of samples per evaluation window.
    :param threshold: Standard-deviation threshold above which the signal is
        considered as varying.
    :return: Boolean array of length ``(len(samples) // window_size) - 1``
        indicating activity in each window.
    """
    n_windows = len(samples) // window_size
    if n_windows < 2:
        # Not enough samples to fill even two windows — treat as non-varying (silent).
        return np.zeros(0, dtype=bool)
    result = np.zeros(n_windows - 1, dtype=bool)
    for i, _ in enumerate(result):
        window = samples[i * window_size : (i + 1) * window_size]
        if abs(np.std(window)) > threshold:
            result[i] = True
    return result
