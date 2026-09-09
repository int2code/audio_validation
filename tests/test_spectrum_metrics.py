"""Unit tests for the windowed spectrum and the THD / THD+N measurements built on it.

The signals here have analytically known answers, so the tests assert accuracy rather
than merely reproducing whatever the implementation currently returns.
"""

import numpy as np
import pytest

from audio_validation.audio_features import (
    AUDIO_BAND_HZ,
    MAX_REPORTED_DISTORTION_PCT,
    AudioFeatures,
    ChannelFeatures,
)
from audio_validation.spectrum import Spectrum

SAMPLE_RATE = 48000
AMPLITUDE = 0.6223  # ~0.44 V RMS, the level the bench actually captures


def _tone(freq_hz, seconds=1.0, amplitude=AMPLITUDE, offset=0.0):
    """Return a sine of *freq_hz* with an optional DC *offset*."""
    samples = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * freq_hz * samples) + offset


def _tone_with_harmonic(freq_hz, thd_percent, seconds=1.0):
    """Return a sine carrying a second harmonic of exactly *thd_percent*."""
    samples = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    fundamental = AMPLITUDE * np.sin(2 * np.pi * freq_hz * samples)
    harmonic = (
        AMPLITUDE * (thd_percent / 100) * np.sin(2 * np.pi * 2 * freq_hz * samples)
    )
    return fundamental + harmonic


# --- Spectrum -------------------------------------------------------------------


def test_amplitude_is_recovered_in_volts_rms():
    """A tone's amplitude reads back in volts, whatever the window costs it."""
    spectrum = Spectrum(_tone(1000.0), SAMPLE_RATE)
    _, amplitude = spectrum.peak_near(1000.0)
    assert amplitude == pytest.approx(AMPLITUDE / np.sqrt(2), rel=1e-3)


def test_amplitude_is_insensitive_to_bin_position():
    """The flat-top window keeps the level flat as the tone moves between bins."""
    levels = [
        Spectrum(_tone(1000 + off), SAMPLE_RATE).peak_near(1000.0)[1]
        for off in (0.0, 0.17, 0.33, 0.5)
    ]
    assert max(levels) / min(levels) == pytest.approx(1.0, abs=2e-3)


def test_peak_near_ignores_a_dc_offset():
    """A large DC offset must not be mistaken for the fundamental."""
    frequency, _ = Spectrum(_tone(100.0, offset=1.0), SAMPLE_RATE).peak_near(100.0)
    assert frequency == pytest.approx(100.0, abs=1.0)


def test_short_buffer_yields_an_empty_spectrum_rather_than_raising():
    """A truncated final chunk degrades to 'cannot measure', not a crash."""
    spectrum = Spectrum(np.array([0.1, -0.1]), SAMPLE_RATE)
    assert spectrum.peak_near(100.0) == (None, 0.0)
    assert spectrum.band_rms(20, 20000) == 0.0


def test_rejects_multichannel_input():
    """Spectrum is a per-channel object; a 2-D array is a caller mistake."""
    with pytest.raises(ValueError):
        Spectrum(np.zeros((100, 2)), SAMPLE_RATE)


# --- THD ------------------------------------------------------------------------


@pytest.mark.parametrize("expected_thd", [0.05, 0.5, 1.0, 2.0])
def test_thd_recovers_a_known_harmonic(expected_thd):
    """An injected second harmonic is measured back at its exact size."""
    spectrum = Spectrum(_tone_with_harmonic(1000.0, expected_thd), SAMPLE_RATE)
    assert ChannelFeatures.calculate_thd(spectrum, 1000.0) == pytest.approx(
        expected_thd, rel=1e-3
    )


@pytest.mark.parametrize("frequency", [100.0, 100.37, 100.5, 999.61])
def test_thd_of_a_clean_tone_is_negligible_off_bin(frequency):
    """Leakage must not be counted as distortion when the tone is not bin-centred.

    Without a window this reads up to 1.1 % on a perfectly clean signal.
    """
    spectrum = Spectrum(_tone(frequency), SAMPLE_RATE)
    assert ChannelFeatures.calculate_thd(spectrum, round(frequency)) < 0.005


@pytest.mark.parametrize("drift_ppm", [0, 100, 1000, 5000])
def test_thd_survives_sample_clock_drift(drift_ppm):
    """Playback/capture clock drift must not manufacture distortion.

    The bench threshold is 0.025 %; unwindowed, ~75 ppm was enough to cross it.
    """
    spectrum = Spectrum(_tone(100.0 * (1 + drift_ppm * 1e-6)), SAMPLE_RATE)
    assert ChannelFeatures.calculate_thd(spectrum, 100.0) < 0.005


def test_thd_uses_the_expected_frequency_not_the_largest_bin():
    """A DC offset larger than the tone must not silently zero the measurement."""
    spectrum = Spectrum(_tone_with_harmonic(100.0, 1.0) + 5.0, SAMPLE_RATE)
    assert ChannelFeatures.calculate_thd(spectrum, 100.0) == pytest.approx(
        1.0, rel=1e-2
    )


def test_thd_counts_harmonics_two_through_five():
    """num_harmonics=5 means H2..H5 — H6 is outside the default window."""
    samples = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    sixth_only = AMPLITUDE * np.sin(
        2 * np.pi * 1000 * samples
    ) + AMPLITUDE * 0.1 * np.sin(2 * np.pi * 6000 * samples)
    spectrum = Spectrum(sixth_only, SAMPLE_RATE)
    assert ChannelFeatures.calculate_thd(spectrum, 1000.0) < 0.01
    assert ChannelFeatures.calculate_thd(
        spectrum, 1000.0, num_harmonics=6
    ) == pytest.approx(10.0, rel=1e-2)


def test_thd_excludes_harmonics_above_nyquist():
    """Harmonics that would alias are dropped, not folded back in."""
    spectrum = Spectrum(_tone(20000.0), SAMPLE_RATE)
    assert ChannelFeatures.calculate_thd(spectrum, 20000.0) < 0.01


# --- THD+N ----------------------------------------------------------------------


@pytest.mark.parametrize("noise_dbc", [-70, -60, -50])
def test_thd_n_recovers_a_known_noise_level(noise_dbc):
    """Broadband noise is measured back, scaled by the 20 Hz-20 kHz band."""
    rms = AMPLITUDE / np.sqrt(2)
    noise_sd = rms * 10 ** (noise_dbc / 20)
    noisy = _tone(1000.0) + np.random.default_rng(0).normal(0, noise_sd, SAMPLE_RATE)
    in_band_fraction = np.sqrt(19980 / (SAMPLE_RATE / 2))
    expected = noise_sd * in_band_fraction / rms * 100
    measured = ChannelFeatures.calculate_thd_n(Spectrum(noisy, SAMPLE_RATE), 1000.0)
    assert measured == pytest.approx(expected, rel=0.1)


def test_thd_n_of_a_clean_tone_is_near_zero():
    """The notch must contain the fundamental, leaving nothing behind."""
    spectrum = Spectrum(_tone(1000.0), SAMPLE_RATE)
    assert ChannelFeatures.calculate_thd_n(spectrum, 1000.0) < 0.01


def test_thd_n_is_at_least_thd_for_the_same_signal():
    """THD+N counts everything THD counts, plus noise."""
    noisy = _tone_with_harmonic(1000.0, 1.0) + np.random.default_rng(1).normal(
        0, 1e-3, SAMPLE_RATE
    )
    spectrum = Spectrum(noisy, SAMPLE_RATE)
    assert ChannelFeatures.calculate_thd_n(
        spectrum, 1000.0
    ) >= ChannelFeatures.calculate_thd(spectrum, 1000.0)


def test_thd_n_ignores_a_dc_offset():
    """DC is removed before the transform, so an offset is not counted as noise."""
    spectrum = Spectrum(_tone(1000.0, offset=2.0), SAMPLE_RATE)
    assert ChannelFeatures.calculate_thd_n(spectrum, 1000.0) < 0.01


# --- degenerate input ------------------------------------------------------------


@pytest.mark.parametrize("metric", ["calculate_thd", "calculate_thd_n"])
def test_silence_reports_none_rather_than_zero(metric):
    """Silence has no fundamental, so distortion is undefined — not 0 %."""
    spectrum = Spectrum(np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    assert getattr(ChannelFeatures, metric)(spectrum, 100.0) is None


def test_a_missing_tone_is_reported_high_but_bounded():
    """A missing fundamental must fail every threshold without breaking the plots."""
    spectrum = Spectrum(_tone(7000.0), SAMPLE_RATE)
    assert (
        ChannelFeatures.calculate_thd_n(spectrum, 100.0) == MAX_REPORTED_DISTORTION_PCT
    )


# --- end to end through AudioFeatures --------------------------------------------


def test_compute_populates_thd_for_every_channel():
    """The wiring from AudioFeatures.compute down to the metrics holds."""
    stereo = np.column_stack([_tone_with_harmonic(100.0, 1.0), _tone(200.0)])
    features = AudioFeatures.compute(
        stereo,
        SAMPLE_RATE,
        expected_frequencies=[[100], [200]],
        tolerance=5.0,
        freq_checker=all,
        activity_threshold=0.05,
    )
    assert features[0].thd_h2_h5 == pytest.approx(1.0, rel=1e-2)
    assert features[1].thd_h2_h5 < 0.005
    assert features[0].thd_audio == pytest.approx(1.0, rel=1e-2)
    assert all(channel.detected for channel in features.channel_features)


def test_multitone_tones_are_resolved_without_spurious_peaks():
    """The flat-top window must still separate the closest multitone pair (7.4 Hz)."""
    tones = [21.3, 28.7, 38.1, 52.4, 70.8, 95.2, 127.6, 171.3, 231.5, 311.2]
    samples = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    multitone = sum(np.sin(2 * np.pi * f * samples) for f in tones) / len(tones)
    features = AudioFeatures.compute(
        multitone.reshape(-1, 1) * 3,
        SAMPLE_RATE,
        expected_frequencies=[tones],
        tolerance=10.0,
        freq_checker=all,
        activity_threshold=0.05,
    )
    assert features[0].detected
    assert features[0].failed_peaks == []
    assert len(features[0].peak_frequencies) == len(tones)


# --- full-band THD ----------------------------------------------------------------


def _tone_with_harmonic_series(freq_hz, harmonics, level_pct):
    """Return a tone carrying each harmonic in *harmonics* at *level_pct* of it."""
    samples = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    signal = AMPLITUDE * np.sin(2 * np.pi * freq_hz * samples)
    for harmonic in harmonics:
        signal += (
            AMPLITUDE
            * (level_pct / 100)
            * np.sin(2 * np.pi * harmonic * freq_hz * samples)
        )
    return signal


def test_thd_audio_counts_harmonics_that_thd_h2_h5_misses():
    """High-order harmonics are invisible to the H2-H5 figure but not to the band one."""
    # four equal harmonics, only one of which is inside H2-H5
    spectrum = Spectrum(
        _tone_with_harmonic_series(100.0, [3, 40, 90, 150], 0.5), SAMPLE_RATE
    )
    low_order = ChannelFeatures.calculate_thd(spectrum, 100.0)
    full_band = ChannelFeatures.calculate_thd_audio(spectrum, 100.0)
    assert low_order == pytest.approx(0.5, rel=1e-2)
    assert full_band == pytest.approx(0.5 * np.sqrt(4), rel=1e-2)


def test_thd_audio_matches_thd_when_all_harmonics_are_low_order():
    """With nothing above H5 the two figures agree."""
    spectrum = Spectrum(_tone_with_harmonic_series(100.0, [2, 3], 1.0), SAMPLE_RATE)
    assert ChannelFeatures.calculate_thd_audio(spectrum, 100.0) == pytest.approx(
        ChannelFeatures.calculate_thd(spectrum, 100.0), rel=1e-3
    )


def test_thd_audio_stops_at_the_top_of_the_band():
    """A harmonic above 20 kHz is excluded even though it is below Nyquist."""
    samples = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    # 21 kHz harmonic of a 7 kHz tone: below Nyquist (24 kHz), above the band
    signal = AMPLITUDE * np.sin(2 * np.pi * 7000 * samples) + AMPLITUDE * 0.01 * np.sin(
        2 * np.pi * 21000 * samples
    )
    spectrum = Spectrum(signal, SAMPLE_RATE)
    assert ChannelFeatures.calculate_thd_audio(spectrum, 7000.0) < 0.05
    assert ChannelFeatures.calculate_thd_audio(
        spectrum, 7000.0, band_hz=(20.0, 24000.0)
    ) == pytest.approx(1.0, rel=1e-2)


def test_thd_audio_is_never_below_thd_h2_h5():
    """The band figure is a superset of the low-order one."""
    spectrum = Spectrum(
        _tone_with_harmonic_series(100.0, [2, 5, 9, 33], 0.2), SAMPLE_RATE
    )
    assert ChannelFeatures.calculate_thd_audio(
        spectrum, 100.0
    ) >= ChannelFeatures.calculate_thd(spectrum, 100.0)


def test_thd_audio_uses_the_shared_audio_band_constant():
    """THD+N and full-band THD must cover the same band to stay comparable."""
    assert AUDIO_BAND_HZ == (20.0, 20000.0)


@pytest.mark.parametrize("metric", ["calculate_thd", "calculate_thd_audio"])
def test_silence_reports_none_for_both_thd_flavours(metric):
    """Neither figure invents a 0 % reading for a signal that is not there."""
    spectrum = Spectrum(np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    assert getattr(ChannelFeatures, metric)(spectrum, 100.0) is None
