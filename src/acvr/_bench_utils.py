"""Shared benchmark/test utilities to avoid duplication.

Functions here are used by both tests and the benchmark script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics
from typing import List

import numpy as np


def decode_frame_number(frame: np.ndarray, channel_index: int) -> int:
    """Decode the embedded frame index in synthetic test fixtures.

    The number is encoded in the top-left 64x2 pixel stripe across channels.
    """

    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    bits = []
    for bit in range(32):
        x0 = 4 + bit * 2
        block = frame[0:2, x0 : x0 + 2, channel_index]
        bits.append(1 if float(block.mean()) > 127.0 else 0)
    value = 0
    for bit, flag in enumerate(bits):
        value |= flag << bit
    return int(value)


def sample_indices(frame_count: int, sample_count: int = 60, seed: int = 0) -> np.ndarray:
    """Return deterministic sample indices."""

    sample_count = min(int(sample_count), int(frame_count))
    rng = np.random.default_rng(int(seed))
    return rng.choice(frame_count, size=sample_count, replace=False)


def error_stats(values: List[int], expected: List[int]) -> tuple[float, int]:
    """Return mean and max absolute error."""

    diffs = [abs(int(v) - int(e)) for v, e in zip(values, expected)]
    return (float(statistics.mean(diffs)) if diffs else 0.0, int(max(diffs) if diffs else 0))


@dataclass(frozen=True)
class AssetData:
    numbers: list[int]
    pts_list: list[int]
    keyframe_pts: list[int]
    time_base: float
    start_pts: int


def load_asset_data(video_path: Path) -> AssetData:
    """Load decoded frame numbers and keyframe metadata via PyAV.

    This function requires that the "av" package is installed.
    """

    import av  # lazy import

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    numbers: list[int] = []
    pts_list: list[int] = []
    keyframe_pts: list[int] = []
    start_pts = int(stream.start_time or 0)
    for frame in container.decode(stream):
        pts = int(frame.pts if frame.pts is not None else (frame.dts if frame.dts is not None else start_pts + len(pts_list)))
        pts_list.append(pts)
        numbers.append(decode_frame_number(frame.to_rgb().to_ndarray(), channel_index=0))
        if bool(getattr(frame, "key_frame", False)):
            keyframe_pts.append(pts)
    container.close()
    return AssetData(
        numbers=numbers,
        pts_list=pts_list,
        keyframe_pts=sorted(set(keyframe_pts)) or [start_pts],
        time_base=float(stream.time_base),
        start_pts=start_pts,
    )

