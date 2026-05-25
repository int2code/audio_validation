"""Audio waveform generation and file output.

This module provides utilities for generating various waveform types and saving them
as multi-channel WAV files. Supports the following waveform shapes:

- Sine wave: pure sinusoidal oscillation at specified frequency
- Square wave: binary waveform oscillating between +1 and -1
- Sawtooth wave: linearly increasing periodic waveform
- White noise: uniformly distributed random noise
- Pink noise: frequency-weighted noise with reduced high-frequency content

Generated audio can be multi-channel, with selective activation of specific channels
and configurable amplitude, sample rate, and duration. Output is saved as 32-bit
PCM WAV files suitable for audio playback and analysis.
"""

import logging
import wave
from typing import Literal, List

import numpy as np

logger = logging.getLogger(__name__)


def _encode_pcm(data: np.ndarray, resolution_bits: int) -> bytes:
    """Encode float32 audio data to raw PCM bytes.

    :param data: float32 array of shape ``(n_samples, n_channels)``
    :param resolution_bits: bit depth – 16, 24, or 32
    :return: raw PCM bytes
    :raises ValueError: if *resolution_bits* is not supported
    """
    if resolution_bits == 16:
        max_val = np.iinfo(np.int16).max
        return (data * max_val).astype(np.int16).tobytes()
    if resolution_bits == 24:
        max_val = 2**23 - 1
        pcm = (data * max_val).astype(np.int32)
        raw = pcm.tobytes()
        return bytes(b for i, b in enumerate(raw) if i % 4 != 3)
    if resolution_bits == 32:
        max_val = np.iinfo(np.int32).max
        return (data * max_val).astype(np.int32).tobytes()
    raise ValueError(
        f"Unsupported resolution_bits: {resolution_bits}. Supported values are 16, 24 and 32."
    )


# pylint:disable=too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-branches
def generate_wave_file(
    shape: Literal["sine", "square", "sawtooth", "white_noise", "pink_noise"],
    freq_list: List[float] = None,
    sample_rate: int = 48000,
    duration=10.0,
    num_channels: int = 24,
    active_channels=None,
    amplitude: float = 0.05,
    resolution_bits: int = 16,
    output_dir: str = "generated_signal.wav",
):
    """Generate a multi-channel waveform and save it as a WAV file.

    Creates a multi-channel audio file with one or more active channels containing
    the specified waveform. Inactive channels contain silence (zeros). The function
    generates the waveform based on shape, frequency, duration, and other parameters,
    then encodes it as 32-bit PCM and saves to a WAV file.

    :param shape: type of waveform to generate
    :param freq_list: fundamental frequency in Hz for each channel
    :param sample_rate: audio sample rate in Hz
    :param duration: duration of the generated audio in seconds
    :param num_channels: total number of channels in the output file
    :param active_channels: list of channel indices to populate with signal;
                           if None, defaults to [0, 1, 2, 3]
    :param amplitude: signal amplitude as a fraction of maximum (range: 0.0-1.0)
    :param output_dir: path to output WAV file

    :raises ValueError: if shape is not a supported waveform type

    Note:
        - Active channels must be within valid range [0, num_channels)
        - Inactive channels contain silence
        - Output is 32-bit PCM mono samples, saved as multi-channel WAV
    """

    logger.info(
        "Generating audio file with %s shape, freq %f Hz, amplitude %.2f, %d sample rate,"
        " %d duration, %d channels, %s active channels and %d bits audio resolution.",
        shape,
        freq_list,
        amplitude,
        sample_rate,
        duration,
        num_channels,
        active_channels,
        resolution_bits,
    )
    if active_channels is None:
        active_channels = [0, 1, 2, 3]

    if freq_list is None:
        freq_list = [400]

    if len(freq_list) != len(active_channels):
        raise ValueError("Length of freq list must match length of active_channels")

    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples)
    data = np.zeros((n_samples, num_channels), dtype=np.float32)

    for i, ch in enumerate(active_channels):
        if 0 <= ch < num_channels:
            current_freq = freq_list[i]

            if shape == "sine":
                signal = np.sin(2 * np.pi * current_freq * t)
            elif shape == "square":
                signal = np.sign(np.sin(2 * np.pi * current_freq * t))
            elif shape == "sawtooth":
                signal = 2 * (t * current_freq - np.floor(0.5 + t * current_freq))
            elif shape == "white_noise":
                signal = np.random.uniform(-1.0, 1.0, size=n_samples)
            elif shape == "pink_noise":
                num_rows = 16
                array = np.random.randn(num_rows, n_samples)
                array = np.cumsum(array, axis=1)
                weights = 2.0 ** (-np.arange(num_rows))
                signal = np.dot(weights, array)
                signal /= np.max(np.abs(signal))
            else:
                raise ValueError(f"Unsupported shape: {shape}.")

            signal = (signal * amplitude).astype(np.float32)
            data[:, ch] = signal

    raw_frames = _encode_pcm(data, resolution_bits)

    # pylint:disable=no-member
    with wave.open(output_dir, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(
            resolution_bits // 8
        )  # Pass resolution in bytes to set the sample width
        wf.setframerate(sample_rate)
        wf.writeframes(raw_frames)


if __name__ == "__main__":
    START_HZ = 100
    STEP_HZ = 0
    CHANNELS = 2
    RESOLUTION_BITS = 16

    generate_wave_file(
        shape="sine",
        freq_list=[START_HZ + i * STEP_HZ for i in range(CHANNELS)],
        sample_rate=48000,
        duration=10,
        num_channels=CHANNELS,
        active_channels=list(range(CHANNELS)),
        amplitude=0.05,
        resolution_bits=RESOLUTION_BITS,
        output_dir=(
            f"{CHANNELS}ch_{RESOLUTION_BITS}bit_freqs"
            f"_start_{START_HZ}hz_step_{STEP_HZ}hz.wav"
        ),
    )
