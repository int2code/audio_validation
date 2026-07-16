"""Bounded rolling audio retention with post-failure front-freeze.

Touched only by the analysis (consumer) thread, so it needs no locking.

Before a failure the buffer evicts the oldest chunks to hold at most
``pre_failure_s`` seconds. On :meth:`freeze` it stops evicting (preserving the
pre-failure window) and keeps appending until ``post_failure_s`` of audio past
the failure has accrued. ``pre_failure_s=None`` keeps everything (memory grows
with run length — caller opt-in).

Retained chunks are stored as :class:`RawChunk` unchanged (original sample
dtype); dtype conversion for the saved WAV happens at write time in the caller.
"""

import logging
from collections import deque
from typing import Deque, Optional

import numpy as np

from audio_validation.continous_validation.models import RawChunk

logger = logging.getLogger(__name__)


class RetentionBuffer:
    """Rolling window of recent raw chunks with front-freeze on failure.

    :param pre_failure_s: Seconds to retain before failure; ``None`` = keep all.
    :param post_failure_s: Seconds to retain after failure (used by callers to
        decide when to stop; enforced by :meth:`post_failure_captured_s`).
    :param sample_rate: Sample rate in Hz (for duration accounting).
    """

    def __init__(
        self,
        pre_failure_s: Optional[float],
        post_failure_s: float,
        sample_rate: int,
    ) -> None:
        self._pre_failure_s = pre_failure_s
        self._post_failure_s = post_failure_s
        self._sample_rate = sample_rate
        self._chunks: Deque[RawChunk] = deque()
        self._frozen = False
        self._failure_end_s: Optional[float] = None

    def add(self, chunk: RawChunk) -> None:
        """Append *chunk*; evict oldest to honour ``pre_failure_s`` unless frozen."""
        self._chunks.append(chunk)
        if self._frozen or self._pre_failure_s is None:
            return
        while len(self._chunks) > 1 and self.total_s() > self._pre_failure_s:
            self._chunks.popleft()

    def freeze(self, failure_end_s: float) -> None:
        """Stop eviction and record the failure boundary time."""
        self._frozen = True
        self._failure_end_s = failure_end_s

    @property
    def frozen(self) -> bool:
        """Whether the front has been frozen (post-failure mode)."""
        return self._frozen

    def post_failure_captured_s(self) -> float:
        """Seconds of audio retained past the failure boundary (0 if not frozen)."""
        if self._failure_end_s is None or not self._chunks:
            return 0.0
        return max(0.0, self._chunks[-1].end_s - self._failure_end_s)

    def total_s(self) -> float:
        """Total retained duration in seconds."""
        if not self._chunks:
            return 0.0
        return self._chunks[-1].end_s - self._chunks[0].start_s

    def start_s(self) -> Optional[float]:
        """Start time of the earliest retained chunk, or ``None`` if empty."""
        return self._chunks[0].start_s if self._chunks else None

    def end_s(self) -> Optional[float]:
        """End time of the latest retained chunk, or ``None`` if empty."""
        return self._chunks[-1].end_s if self._chunks else None

    def concatenate(self) -> Optional[np.ndarray]:
        """Concatenate retained samples to one ``(n, channels)`` array, or ``None``."""
        if not self._chunks:
            return None
        return np.concatenate([chunk.samples for chunk in self._chunks], axis=0)
