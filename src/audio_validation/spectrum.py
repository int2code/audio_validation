"""Windowed single-sided spectrum shared by every frequency-domain measurement.

Every frequency-domain quantity in :mod:`audio_validation.audio_features` — peak
detection, THD, THD+N and level — is read off one :class:`Spectrum` per channel, so
they all see the same window, the same bins and the same scaling.

Two properties matter for accuracy:

**A window is applied.**  Without one (an implicit rectangular window) a tone that does
not complete a whole number of cycles in the analysis buffer leaks across the whole
spectrum, and that leakage is counted as distortion.  A 100 Hz tone in a 1 s buffer at
48 kHz happens to be exactly 100 cycles, so an unwindowed measurement looks correct
until the playback and capture clocks drift apart — at which point a clean signal reads
several tenths of a percent THD.  The default flat-top window makes the reading
insensitive to where the tone falls between bins, at the cost of a wide main lobe.

**Both correction factors are available.**  Windowing attenuates the signal, and the
compensation differs by what is being read: :attr:`amp` carries the amplitude
correction factor and is the array to read a discrete tone's amplitude from, while
:attr:`energy` carries the energy correction factor and is the array to integrate
broadband power over.  Ratio metrics such as THD and THD+N divide two readings taken
from the same array, so the factor cancels and either would do; absolute levels in
volts do not have that luxury.

The conventions match those used by the QA40x analyser software, so results are
directly comparable against the instrument's own readings.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.signal import get_window

#: Window applied before the FFT.  Flat-top trades main-lobe width for amplitude
#: flatness, which is what keeps a tone's measured amplitude independent of where it
#: falls between bins.
DEFAULT_WINDOW = "flattop"

#: Half-width, in Hz, of the default search window used to locate a tone.  Matches the
#: QA40x software's own ``window_Hz_pm``.
DEFAULT_SEARCH_HZ = 10.0


class Spectrum:
    """Single-sided FFT of one channel, windowed and correction-factor scaled.

    :param samples: 1-D array of samples for a single channel, in volts.
    :param sample_rate: Sample rate in Hz.
    :param window: Any window name accepted by :func:`scipy.signal.get_window`.
        Defaults to :data:`DEFAULT_WINDOW`.

    :ivar freqs: 1-D array of bin frequencies in Hz, ``0`` to Nyquist.
    :ivar amp: Amplitude-corrected magnitudes in V RMS — read discrete tone
        amplitudes from this array.
    :ivar energy: Energy-corrected magnitudes in V RMS — integrate band power over
        this array (see :meth:`band_rms`).
    :ivar sample_rate: Sample rate in Hz.
    """

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: int = 48000,
        window: str = DEFAULT_WINDOW,
    ) -> None:
        signal = np.asarray(samples, dtype=np.float64)
        if signal.ndim != 1:
            raise ValueError(f"Spectrum expects a 1-D array, got shape {signal.shape}.")

        self.sample_rate = sample_rate
        self.size = signal.size

        if signal.size < 2:
            # Too short to transform; present an empty spectrum rather than raising so
            # a truncated final chunk degrades to "cannot measure" instead of crashing.
            self.freqs = np.zeros(0)
            self.amp = np.zeros(0)
            self.energy = np.zeros(0)
            return

        # Remove DC before windowing: an offset otherwise both leaks through the window
        # and competes with the fundamental for the largest bin.
        signal = signal - np.mean(signal)
        win = get_window(window, signal.size)

        # rfft gives a true complex half-spectrum; scaling to V RMS per bin.
        magnitude = np.abs(np.fft.rfft(signal * win)) / (signal.size / 2) / np.sqrt(2)
        self.freqs = np.fft.rfftfreq(signal.size, 1 / sample_rate)
        self.amp = magnitude / np.mean(win)
        self.energy = magnitude / np.sqrt(np.mean(win**2))

    @property
    def bin_hz(self) -> float:
        """Width of one FFT bin in Hz."""
        return self.sample_rate / self.size if self.size else 0.0

    @property
    def nyquist(self) -> float:
        """Nyquist frequency in Hz."""
        return self.sample_rate / 2

    def peak_near(
        self, target_hz: float, search_hz: Optional[float] = None
    ) -> Tuple[Optional[float], float]:
        """Locate the largest amplitude bin within *search_hz* of *target_hz*.

        Searching a window around a frequency that is expected, rather than taking the
        largest bin in the whole spectrum, is what keeps a DC offset or an unrelated
        spur from being mistaken for the fundamental.

        The window never reaches DC and never extends below half of *target_hz*, so a
        search for the *n*-th harmonic cannot lock onto the (*n*-1)-th, and a spectrum
        too coarse to resolve the target reports nothing rather than returning an
        arbitrary distant bin.

        :param target_hz: Frequency to search around, in Hz.
        :param search_hz: Half-width of the search window in Hz.  Defaults to
            :data:`DEFAULT_SEARCH_HZ`, widened to four bins when the resolution is
            coarse enough to need it, then clamped to ``target_hz / 2``.
        :return: ``(frequency, amplitude)`` of the largest bin, in Hz and V RMS.
            ``(None, 0.0)`` when no bin falls inside the window.
        """
        if self.freqs.size == 0 or target_hz <= 0:
            return None, 0.0
        if search_hz is None:
            search_hz = max(DEFAULT_SEARCH_HZ, 4 * self.bin_hz)
        search_hz = min(search_hz, target_hz / 2)

        # Skip bin 0: DC is never a tone, and the signal is de-meaned anyway.
        in_window = np.where(np.abs(self.freqs[1:] - target_hz) <= search_hz)[0] + 1
        if in_window.size == 0:
            return None, 0.0

        peak = in_window[np.argmax(self.amp[in_window])]
        return float(self.freqs[peak]), float(self.amp[peak])

    def band_rms(self, f_lo: float, f_hi: float) -> float:
        """Return the RMS of every bin between *f_lo* and *f_hi* inclusive.

        :param f_lo: Lower band edge in Hz.
        :param f_hi: Upper band edge in Hz.
        :return: RMS value in volts; ``0.0`` when the band contains no bins.
        """
        if self.freqs.size == 0 or f_hi <= f_lo:
            return 0.0
        in_band = (self.freqs >= f_lo) & (self.freqs <= f_hi)
        return float(np.sqrt(np.sum(self.energy[in_band] ** 2)))

    def normalised_amplitudes(self) -> np.ndarray:
        """Return :attr:`amp` scaled so its largest value is ``1.0``.

        Used for peak detection, where the thresholds are expressed relative to the
        strongest component rather than in volts.

        :return: 1-D array of normalised amplitudes; all zeros for a silent channel.
        """
        if self.amp.size == 0:
            return self.amp
        peak = np.max(self.amp)
        return self.amp / peak if peak > 0 else self.amp
