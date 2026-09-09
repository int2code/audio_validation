### 0.5.0

[MINOR] Report THD over the whole audio band as well as H2-H5.

A single THD figure cannot describe a device whose distortion runs to high order. On the
bench DUT the 2nd-5th harmonics carry only ~19 % of the harmonic power: the rest sits in
a series of odd-order harmonics reaching past 18 kHz, which the classic figure does not
see at all.

- `ChannelFeatures.thd` is renamed **`thd_h2_h5`**, so the harmonic range is part of the
  name rather than something the reader has to look up.
- New **`thd_audio`** counts every harmonic inside `AUDIO_BAND_HZ` (20 Hz - 20 kHz), the
  same band as `thd_n`, so the two are directly comparable: when they agree the residual
  is all harmonic, and when THD+N is much larger there is real broadband noise underneath.
  On the bench DUT they land within 3 % of each other.
- `ChannelFeatures.calculate_thd_audio` is the new entry point;
  `calculate_thd` is unchanged except for taking its default harmonic count from
  `DEFAULT_THD_HARMONICS`.
- `AudioCriteria.max_thd_audio` and `check_thd_audio` gate the new figure. Left at `None`
  it is recorded and plotted but not judged.
- `ChannelMetric.thd` is renamed to match, the metrics CSV gains a `thd_audio` column
  between `thd_h2_h5` and `thd_n`, and the timeline plot gains a fourth series.

Breaking: `ChannelFeatures.thd` and `ChannelMetric.thd` are now `thd_h2_h5`, and the
metrics CSV column `thd` is now `thd_h2_h5`.

### 0.4.0

[MINOR] Measure THD and THD+N from a windowed spectrum.

Every frequency-domain measurement now reads off one `Spectrum` per channel, built
with a flat-top window and both correction factors. Previously the spectrum was
unwindowed, so a tone that did not complete a whole number of cycles in the analysis
buffer leaked across the spectrum and the leakage was counted as distortion — a clean
signal read up to 1.1 % THD, and ~75 ppm of playback/capture clock drift was enough to
cross a 0.025 % threshold.

- THD and THD+N locate the fundamental near the *expected* frequency instead of taking
  the largest bin, so a DC offset can no longer be mistaken for it.
- THD+N notches +-0.5 octave around the fundamental and measures over 20 Hz - 20 kHz,
  matching the QA40x conventions; the previous fixed +-8-bin notch could not contain
  the fundamental at buffer lengths where the tone fell between bins.
- Uses `numpy.fft.rfft` rather than `scipy.fftpack.rfft`, whose packed real layout was
  being read as a magnitude spectrum. Bin-width arithmetic was consequently off by 2x
  and peak amplitudes varied with the tone's phase.
- Both metrics return `None` when the fundamental is absent, and are capped at
  `MAX_REPORTED_DISTORTION_PCT`.

Breaking: `ChannelFeatures.calculate_thd` and `ChannelFeatures.calculate_thd_n` now
take a `Spectrum` and the expected fundamental in Hz.

### 0.3.7

[PATCH] Fix THD+N calculation in ChannelFeatures

### 0.3.6

[PATCH] Skip evaluating partial chunks.

### 0.3.5

[PATCH] Add saving metrics to csv file during the validation run.

### 0.3.4

[PATCH] Add non-blocking chunk validation skip to transition conditions.

### 0.3.3

[PATCH] Add absolute timestamps in audio chunks.

### 0.3.2

[PATCH] Add warmup chunks, improve failure reporting.

### 0.3.1

[PATCH] Fix printing Continous Audio Validator Failure Info.

### 0.3.0

[MINOR] Add Continous Audio Validator.

### 0.2.0

[MINOR] Add THD, THD+N calculation.

### 0.1.2

[MINOR] Initial release.