# audio_validation

A Python library for validating and analysing audio data, including feature extraction, bit-exact verification against a reference, and waveform generation utilities.

## Features

- **Feature extraction** — compute RMS, peak/min/max, mean, FFT-based frequency detection, and per-channel audio onset offset for multi-channel captures.
- **Audio verification** — validate a captured audio stream against a reference WAV file with drift-tolerant re-synchronisation, detailed mismatch reporting, and automatic artefact generation (PNG plots and WAV snippets).
- **Waveform generation** — generate multi-channel WAV files with configurable waveforms: sine, square, sawtooth, white noise, and pink noise; supports 16-, 24-, and 32-bit PCM output.

## Requirements

- Python ≥ 3.11
- See [requirements.txt](requirements.txt) for runtime dependencies (`numpy`, `scipy`, `sounddevice`, `pymodbus`, …).

## Installation

```bash
pip install audio_validation
```

Or, for development:

```bash
pip install -e ".[dev]"
```

## Usage

### Feature extraction

```python
from audio_validation.audio_features import AudioFeatures

features = AudioFeatures.compute(
    samples=raw_samples,           # numpy array, shape (n_samples, n_channels)
    sample_rate=48000,
    expected_frequencies=[400, 800],
    tolerance=50,
)

for ch_idx, ch in enumerate(features.channels):
    print(f"Ch {ch_idx}: detected={ch.detected}, rms={ch.rms:.4f}, peaks={ch.peak_frequencies}")
```

### Audio verification

```python
from audio_validation.audio_verification import verify_audio

results = verify_audio(
    reference_path="reference.wav",
    detected_samples=captured_array,
    sample_rate=48000,
    artifacts_dir="test_artifacts/",
)
```

### Waveform generation

```python
from audio_validation.utils.audio_generation import generate_wave_file

generate_wave_file(
    shape="sine",
    freq_list=[1000.0],
    sample_rate=48000,
    duration=5.0,
    num_channels=2,
    active_channels=[0, 1],
    amplitude=0.05,
    resolution_bits=16,
    output_dir="output.wav",
)
```

## Maintainers

- Hubert Stepniewski — hubert.stepniewski@int2code.com
- Marcin Tomiczek — marcin.tomiczek@int2code.com
- Piotr Sznapka — piotr.sznapka@int2code.com

## License

See [LICENSE](LICENSE).
