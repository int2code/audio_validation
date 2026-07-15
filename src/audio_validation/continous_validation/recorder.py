"""Streaming capture abstraction for continuous analysis.

The analyser pulls fixed-size contiguous blocks from a :class:`Recorder`. A true
gapless recorder keeps its capture running continuously between reads (e.g. the
QA40x streaming recorder in ``muraba_tb``). :class:`CallableRecorder` adapts a
plain blocking ``capture_fn(duration_s) -> ndarray`` and is therefore
**not gapless** — a small gap exists between consecutive calls.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Recorder(Protocol):
    """Streaming capture source.

    Implementations must  return **contiguous** samples across consecutive :meth:`read_capture` calls.
    """

    def start_capture(self) -> None:
        """Begin continuous capture."""

    def read_capture(self, n_samples: int) -> np.ndarray:
        """Return up to *n_samples* new samples, shaped ``(m, channels)``.

        Blocks until data is available. Returns fewer than *n_samples* (possibly
        empty) only when the stream is ending/stopped; an empty return signals
        end-of-stream.
        """

    def stop_capture(self) -> None:
        """Stop capture and release resources."""
