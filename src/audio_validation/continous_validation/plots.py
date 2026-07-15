"""Metrics-timeline plotting for continuous audio validation.

Renders per-channel RMS / THD / THD+N over the run as a finalisation artifact,
with an optional vertical marker at the failure time.
"""

import logging
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe backend
from matplotlib import pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

from audio_validation.continous_validation.models import ChunkMetrics

logger = logging.getLogger(__name__)

_SERIES = [("rms", "RMS"), ("thd", "THD %"), ("thd_n", "THD+N %")]


def plot_metrics_timeline(
    metrics: List[ChunkMetrics],
    out_path: str,
    failure_time_s: Optional[float] = None,
) -> None:
    """Render RMS / THD / THD+N vs time, one line per channel, to *out_path*.

    No-op when *metrics* is empty.

    :param metrics: Per-chunk metrics timeline.
    :param out_path: Destination PNG path.
    :param failure_time_s: If set, draw a vertical marker at this time (seconds).
    """
    if not metrics:
        logger.info("No metrics to plot; skipping timeline plot.")
        return

    n_channels = max((len(m.channels) for m in metrics), default=0)
    times = [m.end_s for m in metrics]

    fig, axes = plt.subplots(len(_SERIES), 1, figsize=(12, 9), sharex=True)
    for ax, (attr, label) in zip(axes, _SERIES):
        for ch in range(n_channels):
            values = [
                getattr(m.channels[ch], attr) if ch < len(m.channels) else None
                for m in metrics
            ]
            ax.plot(times, values, marker=".", label=f"ch{ch}")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        if failure_time_s is not None:
            ax.axvline(failure_time_s, color="red", linestyle="--", label="failure")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Continuous validation metrics timeline")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved metrics timeline plot: %s", out_path)
