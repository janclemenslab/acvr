"""Smoke tests for benchmark runner integration."""

from pathlib import Path

import pytest

from acvr.benchmarks import BenchmarkConfig, run_benchmark_suite


def test_benchmark_runner_smoke() -> None:
    """Ensure benchmark runner can execute on a fixture video."""

    video_path = Path(__file__).resolve().parent / "data" / "test_cfr_h264.mp4"
    if not video_path.exists():
        pytest.skip(f"Missing test video: {video_path}")

    config = BenchmarkConfig(
        metric="truth",
        samples=5,
        index_pattern="sequential",
        compare_fastvideo=False,
        no_opencv=True,
    )
    results = run_benchmark_suite([video_path], config)
    assert results
    result = results[0]
    assert result.timing_ms["fast"] is not None
    assert result.timing_ms["accurate"] is not None
