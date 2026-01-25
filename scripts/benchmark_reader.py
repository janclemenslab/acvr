"""Benchmark read performance for acvr modes."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from acvr import VideoReader


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the benchmark."""

    parser = argparse.ArgumentParser(description="Benchmark acvr read modes.")
    parser.add_argument("path", type=Path, help="Path to video file.")
    parser.add_argument("--samples", type=int, default=100, help="Number of frames to sample.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sample indices.")
    return parser.parse_args()


def time_reads(label: str, func, indices: np.ndarray) -> None:
    """Time a read function over a set of indices."""

    timings = []
    failures = 0
    for idx in indices:
        start = time.perf_counter()
        try:
            func(int(idx))
        except Exception as exc:
            failures += 1
            print(f"{label:>12}: failed ({exc})")
            continue
        timings.append(time.perf_counter() - start)

    if not timings:
        return

    median = statistics.median(timings)
    fps = 1.0 / median if median > 0 else float("inf")
    failure_msg = f" | failed: {failures}/{len(indices)}" if failures else ""
    print(f"{label:>12}: {median * 1000:.2f} ms/frame ({fps:.1f} fps){failure_msg}")


def accuracy_report(
    label: str,
    values: list[int],
    expected: list[int],
    *,
    failures: int = 0,
) -> None:
    """Report accuracy for decoded frame indices."""

    failures = max(failures, sum(1 for v in values if v < 0))
    matched = [(v, e) for v, e in zip(values, expected) if v >= 0]
    diffs = [abs(v - e) for v, e in matched]
    mean_err = statistics.mean(diffs) if diffs else float("nan")
    max_err = max(diffs) if diffs else float("nan")
    within_1 = sum(1 for d in diffs if d <= 1)
    within_2 = sum(1 for d in diffs if d <= 2)
    total = len(expected)
    print(
        f"{label:>12}: mean {mean_err:.2f} | max {max_err} | <=1: {within_1}/{len(diffs)}"
        f" | <=2: {within_2}/{len(diffs)} | failed: {failures}/{total}"
    )


def time_opencv(path: Path, indices: np.ndarray) -> None:
    """Benchmark OpenCV frame access by index."""

    try:
        import cv2
    except ImportError:
        print(f"{'opencv':>12}: skipped (cv2 not installed)")
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"{'opencv':>12}: failed to open")
        return

    timings = []
    for idx in indices:
        start = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        cap.read()
        timings.append(time.perf_counter() - start)
    cap.release()

    median = statistics.median(timings)
    fps = 1.0 / median if median > 0 else float("inf")
    print(f"{'opencv':>12}: {median * 1000:.2f} ms/frame ({fps:.1f} fps)")


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


def accuracy_benchmark(reader: VideoReader, indices: np.ndarray, frame_rate: float) -> None:
    """Benchmark accuracy for each read mode."""

    expected = [int(idx) for idx in indices]

    accurate = [decode_frame_number(reader[int(idx)], channel_index=0) for idx in indices]
    accuracy_report("accurate", accurate, expected)

    scrub = [
        decode_frame_number(reader.read_keyframe_at(int(idx) / frame_rate).image, channel_index=0)
        for idx in indices
    ]
    accuracy_report("scrub", scrub, expected)

    fast = []
    fast_failures = 0
    for idx in indices:
        try:
            frame = reader.read_frame_fast(index=int(idx), decode_rgb=True).image
        except Exception:
            fast_failures += 1
            fast.append(-1)
        else:
            fast.append(decode_frame_number(frame, channel_index=0))
    accuracy_report("fast", fast, expected, failures=fast_failures)

    fast_rgb = []
    fast_rgb_failures = 0
    for idx in indices:
        try:
            frame = reader.read_frame_fast(index=int(idx), decode_rgb=True).image
        except Exception:
            fast_rgb_failures += 1
            fast_rgb.append(-1)
        else:
            fast_rgb.append(decode_frame_number(frame, channel_index=0))
    accuracy_report("fast_rgb", fast_rgb, expected, failures=fast_rgb_failures)

    try:
        import cv2
    except ImportError:
        print(f"{'opencv':>12}: skipped (cv2 not installed)")
        return

    cap = cv2.VideoCapture(reader._backend._path)
    if not cap.isOpened():
        print(f"{'opencv':>12}: failed to open")
        return

    opencv_vals = []
    opencv_failures = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            opencv_failures += 1
            opencv_vals.append(-1)
        else:
            opencv_vals.append(decode_frame_number(frame, channel_index=2))
    cap.release()
    accuracy_report("opencv", opencv_vals, expected, failures=opencv_failures)


def main() -> None:
    """Run the benchmark for accurate, scrub, and fast modes."""

    args = parse_args()
    if not args.path.exists():
        raise SystemExit(f"Missing video file: {args.path}")

    with VideoReader(str(args.path)) as reader:
        frame_count = reader.number_of_frames
        if frame_count <= 0:
            raise SystemExit("Video reports zero frames.")

        sample_count = min(args.samples, frame_count)
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(frame_count, size=sample_count, replace=False)

        reader.build_keyframe_index()
        frame_rate = reader.frame_rate or 1.0

        print(f"Frames: {frame_count} | Samples: {sample_count}")

        time_reads("accurate", lambda idx: reader[idx], indices)
        time_reads("scrub", lambda idx: reader.read_keyframe_at(idx / frame_rate), indices)
        time_reads("fast", lambda idx: reader.read_frame_fast(index=idx), indices)
        time_reads(
            "fast_rgb",
            lambda idx: reader.read_frame_fast(index=idx, decode_rgb=True),
            indices,
        )
        time_opencv(args.path, indices)
        print("\nAccuracy (frame index error):")
        accuracy_benchmark(reader, indices, frame_rate)


if __name__ == "__main__":
    main()
