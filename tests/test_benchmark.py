"""Benchmark-style tests for accuracy and relative performance."""

from __future__ import annotations

from pathlib import Path

import pytest

from acvr.benchmarks import BenchmarkConfig, run_benchmark_suite


@pytest.mark.parametrize("asset_name", ["test_cfr_h264.mp4", "test_vfr_h264.mp4"])
def test_accuracy_benchmarks(asset_name: str) -> None:
    """Validate accuracy against embedded frame numbers."""

    video_path = Path(__file__).resolve().parent / "data" / asset_name
    if not video_path.exists():
        pytest.skip(f"Missing test video: {video_path}")

    config = BenchmarkConfig(
        metric="embedded",
        samples=60,
        index_pattern="random",
        compare_fastvideo=False,
        no_opencv=True,
    )
    results = run_benchmark_suite([video_path], config)
    if not results:
        pytest.skip("Video reports no frames.")
    acc = results[0].accuracy or {}
    accurate = acc.get("accurate")
    scrub = acc.get("scrub")
    fast = acc.get("fast")
    assert accurate is not None
    assert scrub is not None
    assert fast is not None
    assert accurate["mean"] == 0.0
    assert accurate["max"] == 0
    assert fast["mean"] <= scrub["mean"]
    assert fast["max"] <= scrub["max"]


@pytest.mark.parametrize("asset_name", ["test_cfr_h264.mp4", "test_vfr_h264.mp4"])
def test_fast_sequential_accuracy(asset_name: str) -> None:
    """Ensure fast sequential reads follow timeline-based indexing."""

    video_path = Path(__file__).resolve().parent / "data" / asset_name
    if not video_path.exists():
        pytest.skip(f"Missing test video: {video_path}")

    config = BenchmarkConfig(
        metric="truth",
        samples=60,
        index_pattern="sequential",
        compare_fastvideo=False,
        no_opencv=True,
    )
    results = run_benchmark_suite([video_path], config)
    if not results:
        pytest.skip("Video reports no frames.")
    acc = results[0].accuracy or {}
    fast = acc.get("fast")
    assert fast is not None
    assert fast["mean"] == 0.0
    assert fast["max"] == 0


def test_speed_benchmarks() -> None:
    """Ensure relative speed ordering remains stable."""

    video_path = Path(__file__).resolve().parent / "data" / "test_cfr_h264.mp4"
    if not video_path.exists():
        pytest.skip(f"Missing test video: {video_path}")

    config = BenchmarkConfig(
        metric="truth",
        samples=40,
        index_pattern="random",
        compare_fastvideo=False,
        no_opencv=True,
    )
    results = run_benchmark_suite([video_path], config)
    if not results:
        pytest.skip("Video reports no frames.")
    timing = results[0].timing_ms
    scrub_time = timing["scrub"] or 0.0
    fast_time = timing["fast"] or 0.0
    accurate_time = timing["accurate"] or 0.0
    assert scrub_time <= fast_time * 1.2
    assert fast_time <= accurate_time * 1.2
