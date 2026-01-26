"""Benchmark-style tests for accuracy and relative performance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics
import time

import numpy as np
import pytest

from acvr import VideoReader


@dataclass(frozen=True)
class AssetData:
    numbers: list[int]
    pts_list: list[int]
    keyframe_pts: list[int]
    time_base: float
    start_pts: int


def decode_frame_number(frame: np.ndarray, channel_index: int) -> int:
    """Decode the embedded frame index in the test fixtures."""

    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    bits = []
    for bit in range(32):
        x0 = 4 + bit * 2
        block = frame[0:2, x0 : x0 + 2, channel_index]
        bits.append(1 if block.mean() > 127 else 0)
    value = 0
    for bit, flag in enumerate(bits):
        value |= flag << bit
    return value


def load_asset_data(video_path: Path) -> AssetData:
    """Load decoded frame numbers and keyframe metadata."""

    av = pytest.importorskip("av")
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    numbers: list[int] = []
    pts_list: list[int] = []
    keyframe_pts: list[int] = []
    start_pts = stream.start_time or 0
    for frame in container.decode(stream):
        pts = frame.pts if frame.pts is not None else frame.dts
        if pts is None:
            pts = start_pts + len(pts_list)
        pts = int(pts)
        pts_list.append(pts)
        numbers.append(decode_frame_number(frame.to_rgb().to_ndarray(), channel_index=0))
        if frame.key_frame:
            keyframe_pts.append(pts)
    container.close()
    return AssetData(
        numbers=numbers,
        pts_list=pts_list,
        keyframe_pts=sorted(set(keyframe_pts)) or [start_pts],
        time_base=float(stream.time_base),
        start_pts=start_pts,
    )


def sample_indices(frame_count: int, sample_count: int = 60, seed: int = 0) -> np.ndarray:
    """Return deterministic sample indices."""

    sample_count = min(sample_count, frame_count)
    rng = np.random.default_rng(seed)
    return rng.choice(frame_count, size=sample_count, replace=False)


def expected_keyframe_numbers(
    indices: np.ndarray,
    asset: AssetData,
    frame_rate: float,
) -> list[int]:
    """Return expected keyframe frame numbers for scrub reads."""

    pts_to_index = {pts: idx for idx, pts in enumerate(asset.pts_list)}
    keyframes = asset.keyframe_pts
    if not keyframes:
        return [asset.numbers[idx] for idx in indices]

    expected = []
    for idx in indices:
        t_s = int(idx) / frame_rate
        target_pts = asset.start_pts + int(round(t_s / asset.time_base))
        pos = np.searchsorted(keyframes, target_pts, side="right") - 1
        pos = max(0, min(pos, len(keyframes) - 1))
        kf_pts = keyframes[pos]
        frame_idx = pts_to_index.get(kf_pts, 0)
        expected.append(asset.numbers[frame_idx])
    return expected


def time_reads(func, indices: np.ndarray) -> float:
    """Return the median elapsed time for a read function."""

    timings = []
    for idx in indices:
        start = time.perf_counter()
        func(int(idx))
        timings.append(time.perf_counter() - start)
    return statistics.median(timings) if timings else 0.0


def error_stats(values: list[int], expected: list[int]) -> tuple[float, int]:
    """Return mean and max absolute error."""

    diffs = [abs(v - e) for v, e in zip(values, expected)]
    return (statistics.mean(diffs) if diffs else 0.0, max(diffs) if diffs else 0)


@pytest.mark.parametrize("asset_name", ["test_cfr_h264.mp4", "test_vfr_h264.mp4"])
def test_accuracy_benchmarks(asset_name: str) -> None:
    """Validate accuracy against embedded frame numbers."""

    video_path = Path(__file__).resolve().parent / "data" / asset_name
    if not video_path.exists():
        pytest.skip(f"Missing test video: {video_path}")

    asset = load_asset_data(video_path)
    if not asset.numbers:
        pytest.skip("Video reports no frames.")

    with VideoReader(str(video_path)) as reader:
        reader.build_keyframe_index()
        frame_rate = reader.frame_rate or 1.0
        frame_count = min(reader.number_of_frames, len(asset.numbers))
        indices = sample_indices(frame_count)

        accurate = [decode_frame_number(reader[int(idx)], channel_index=0) for idx in indices]
        expected = [asset.numbers[int(idx)] for idx in indices]
        assert accurate == expected

        scrub = [
            decode_frame_number(
                reader.read_keyframe_at(int(idx) / frame_rate).image,
                channel_index=0,
            )
            for idx in indices
        ]
        expected_scrub = expected_keyframe_numbers(indices, asset, frame_rate)
        assert scrub == expected_scrub

        fast = [
            decode_frame_number(
                reader.read_frame_fast(index=int(idx), decode_rgb=True).image,
                channel_index=0,
            )
            for idx in indices
        ]
        mean_fast, max_fast = error_stats(fast, expected)
        mean_scrub, max_scrub = error_stats(scrub, expected)
        assert mean_fast <= mean_scrub
        assert max_fast <= max_scrub


def test_speed_benchmarks() -> None:
    """Ensure relative speed ordering remains stable."""

    video_path = Path(__file__).resolve().parent / "data" / "test_cfr_h264.mp4"
    if not video_path.exists():
        pytest.skip(f"Missing test video: {video_path}")

    with VideoReader(str(video_path)) as reader:
        reader.build_keyframe_index()
        frame_rate = reader.frame_rate or 1.0
        frame_count = min(reader.number_of_frames, 200)
        indices = sample_indices(frame_count, sample_count=40, seed=1)

        accurate_time = time_reads(lambda idx: reader[idx], indices)
        scrub_time = time_reads(lambda idx: reader.read_keyframe_at(idx / frame_rate), indices)
        fast_time = time_reads(lambda idx: reader.read_frame_fast(index=idx), indices)

    assert scrub_time <= fast_time * 1.2
    assert fast_time <= accurate_time * 1.2

    try:
        import cv2
    except ImportError:
        return

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    timings = []
    for idx in indices:
        start = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        cap.read()
        timings.append(time.perf_counter() - start)
    cap.release()

    opencv_time = statistics.median(timings) if timings else 0.0
    assert opencv_time <= accurate_time * 1.5
