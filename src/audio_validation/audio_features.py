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
from scipy.fftpack import rfft, rfftfreq
from scipy.signal import find_peaks
from scipy.io import wavfile

logger = logging.getLogger(__name__)
save_plot_lock = threading.Lock()


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
    :cvar thd: Total Harmonic Distortion as a percentage (e.g. ``1.0`` for 1 % THD).
        Populated when FFT detection is requested; ``0.0`` otherwise.
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
    thd: float = 0.0
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
        :return: Fully populated :class:`ChannelFeatures` instance.  :attr:`thd`
            is computed from the full FFT spectrum when FFT detection is
            requested; ``0.0`` otherwise.
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
        thd_val = 0.0

        if (
            expected_frequencies is not None
            and tolerance is not None
            and freq_checker is not None
        ):
            expected_freq_approx = _calculate_approx_values(
                expected_frequencies, tolerance
            )
            x_frequencies, y_amplitudes = ChannelFeatures._full_spectrum(
                samples, sample_rate
            )
            peak_frequencies, peak_amplitudes = ChannelFeatures._peaks_from_spectrum(
                x_frequencies, y_amplitudes
            )
            thd_val = ChannelFeatures.calculate_thd(
                x_frequencies, y_amplitudes, sample_rate
            )
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
            thd=thd_val,
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
        """Compute the normalised RFFT spectrum of a 1-D sample array.

        :param samples: 1-D array of samples for a single channel.
        :param sample_rate: Sample rate in Hz.
        :return: Tuple ``(frequencies, amplitudes)`` — two 1-D arrays covering
            **all** RFFT bins.  Amplitudes are normalised so the maximum value
            is ``1.0``.
        """
        y_amplitudes = np.abs(rfft(samples))
        y_amplitudes /= np.max(y_amplitudes)
        x_frequencies = rfftfreq(samples.size, 1 / sample_rate)
        return x_frequencies, y_amplitudes

    @staticmethod
    def _peaks_from_spectrum(x_frequencies, y_amplitudes, **find_peaks_kwargs):
        """Extract RFFT peaks from a precomputed spectrum.

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
        """Calculate RFFT peaks from a 1-D sample array.

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
    def calculate_thd(
        freqs: np.ndarray,
        amplitudes: np.ndarray,
        samplerate: int,
        num_harmonics: int = 5,
    ) -> float:
        """Calculate Total Harmonic Distortion (THD) from a full FFT spectrum.

        Uses the power-ratio definition::

            THD = sqrt(P2 + P3 + ... + Pn) / P1 * 100 %

        where ``P1`` is the power at the fundamental frequency and ``P2``–``Pn``
        are the powers at the 2nd through *num_harmonics*-th harmonics.  Energy
        at each harmonic is summed over a narrow bin window to account for
        spectral leakage (see :meth:`_get_freq_energy`).  Harmonics above the
        Nyquist frequency are excluded automatically.

        :param freqs: 1-D array of FFT bin frequencies in Hz — **all bins**, not
            just detected peaks (e.g. as returned by
            :func:`scipy.fftpack.rfftfreq`).
        :param amplitudes: 1-D array of normalised FFT amplitudes corresponding
            to *freqs*; should be normalised so the maximum value is ``1.0``.
        :param samplerate: Sample rate in Hz; used to exclude harmonics above
            the Nyquist frequency.
        :param num_harmonics: Number of harmonics above the fundamental to
            include (default ``5``, covering harmonics 2–6).
        :return: THD as a percentage (e.g. ``1.0`` for 1 % THD).  Returns
            ``0.0`` when the fundamental frequency or its power is zero.
        """
        fund_freq = freqs[np.argmax(amplitudes)]

        if fund_freq <= 0:
            return 0.0

        fund_power = ChannelFeatures._get_freq_energy(
            freqs, amplitudes, fund_freq, samplerate
        )
        if fund_power == 0:
            return 0.0

        harmonics_power = sum(
            ChannelFeatures._get_freq_energy(
                freqs, amplitudes, n * fund_freq, samplerate
            )
            for n in range(2, num_harmonics + 1)
        )

        return np.sqrt(harmonics_power / fund_power) * 100

    @staticmethod
    def _get_freq_energy(
        freqs: np.ndarray, amplitude: np.ndarray, harm_freq: float, samplerate: int
    ) -> float:
        """Return the summed squared amplitude in a 7-bin window centred on *harm_freq*.

        FFT windowing spreads the energy of a pure tone across several adjacent
        bins.  Summing a narrow neighbourhood captures the total power of that
        harmonic more accurately than reading a single bin.  Returns ``0.0``
        immediately when *harm_freq* is at or above the Nyquist frequency.

        :param freqs: 1-D array of FFT bin frequencies in Hz (all bins).
        :param amplitude: 1-D array of normalised FFT amplitudes corresponding
            to *freqs*.
        :param harm_freq: Target frequency in Hz (typically a harmonic of the
            fundamental).
        :param samplerate: Sample rate in Hz — used to compute the Nyquist limit.
        :return: Sum of squared amplitudes in the ±3-bin window around the
            closest bin to *harm_freq*; ``0.0`` if above Nyquist.
        """
        # Avoid calculating harmonics above Nyquist frequency.
        if harm_freq >= samplerate / 2:
            return 0.0

        idx = np.argmin(np.abs(freqs - harm_freq))
        lower = max(0, idx - 3)
        upper = min(amplitude.size - 1, idx + 4)

        return np.sum(amplitude[lower:upper] ** 2)


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
                    ch_samples, sample_rate, threshold=activity_threshold)
                if pre_trim_offset_s >= 0:
                    ch_samples = ch_samples[int(
                        pre_trim_offset_s * sample_rate):]
                else:
                    logger.debug(
                        "skip_latency: channel %d is entirely silent, no trimming.", ch
                    )
            elif skip_first > 0:
                ch_samples = ch_samples[skip_first:]

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
        filepath: str, channels: int = 1, skip_first: int = 0
    ) -> "AudioFeatures":
        """Build :class:`AudioFeatures` from a WAV file (no FFT detection).

        :param filepath: Path to the WAV file.
        :param channels: Number of channels to load (first *channels* tracks).
        :param skip_first: Samples to discard from the front of every channel.
        :return: :class:`AudioFeatures` with basic statistics per channel;
            ``detected`` is ``False`` on every :class:`ChannelFeatures`.
        """
        _, data = wavfile.read(filepath)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        wav_samples = data[:, :channels]
        if skip_first:
            wav_samples = wav_samples[skip_first:]
        return AudioFeatures.compute(samples=wav_samples)


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
        window = samples[i * window_size: (i + 1) * window_size]
        if abs(np.std(window)) > threshold:
            result[i] = True
    return result
