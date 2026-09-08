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