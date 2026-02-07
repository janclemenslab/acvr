"""Public package interface for acvr."""

from .benchmarks import BenchmarkConfig, run_benchmark_suite
from .reader import VideoReader

__all__ = ["BenchmarkConfig", "VideoReader", "run_benchmark_suite"]
__version__ = "0.1.0"
