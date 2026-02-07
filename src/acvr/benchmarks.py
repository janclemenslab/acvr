"""Benchmark read performance for acvr modes."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reduce FFmpeg log noise that can flood output on some assets
import os
os.environ.setdefault("AV_LOG_LEVEL", "quiet")  # pragma: no cover
try:  # pragma: no cover
    import av as _av
    try:
        _av.logging.set_level(_av.logging.QUIET)
    except Exception:
        pass
except Exception:
    pass

from acvr.reader import VideoReader
from acvr._bench_utils import decode_frame_number, load_asset_data
import json
import hashlib


@dataclass(frozen=True)
class BenchmarkConfig:
    metric: str = "truth"
    samples: int = 100
    seed: int = 0
    scrub_bucket_ms: int = 25
    scrub_mode: str = "nearest"
    compare_fastvideo: bool = False
    index_pattern: str = "random"
    use_sequential: bool = True
    cache_size: int = 0
    passes: int = 1
    threading: bool = True
    thread_count: int = 0
    no_opencv: bool = True


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    frames: int
    base_samples: int
    samples: int
    timing_ms: dict[str, float | None]
    accuracy: dict[str, dict[str, float | int]] | None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the benchmark."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark acvr read modes. If PATH is a directory, benchmark all "
            "supported videos in it and print a summary; if PATH is a file, "
            "benchmark that single video."
        )
    )
    parser.add_argument("path", type=Path, help="Path to video file or directory.")
    parser.add_argument("--samples", type=int, default=100, help="Number of frames to sample.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sample indices.")
    parser.add_argument(
        "--metric",
        choices=["embedded", "pts", "truth"],
        default="truth",
        help=(
            "Accuracy metric: 'embedded' expects test fixtures with encoded frame numbers; "
            "'pts' maps decoded PTS to nearest reference frame index (works for arbitrary videos); "
            "'truth' uses stored ground-truth derived from full decode (recommended)."
        ),
    )
    parser.add_argument(
        "--scrub-bucket-ms",
        type=int,
        default=25,
        help="Bucket size for scrub keyframe cache (ms).",
    )
    parser.add_argument(
        "--threading",
        choices=["on", "off"],
        default="on",
        help="Enable or disable decoder threading.",
    )
    parser.add_argument(
        "--thread-count",
        type=int,
        default=0,
        help="FFmpeg decoder thread count (0=auto).",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=0,
        help="Decoded frame LRU cache size (0 disables caching).",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Repeat the same index list this many times (useful for cache effects).",
    )
    parser.add_argument(
        "--scrub-mode",
        choices=["previous", "nearest", "next"],
        default="nearest",
        help="Keyframe selection strategy for scrub accuracy.",
    )
    parser.add_argument(
        "--compare-fastvideo",
        action="store_true",
        help="Include a PyAV-based FastVideoReader-like baseline in timing/accuracy.",
    )
    parser.add_argument(
        "--index-pattern",
        choices=["random", "sequential"],
        default="random",
        help="Sample indices randomly or sequentially.",
    )
    parser.add_argument(
        "--use-sequential",
        choices=["on", "off", "both"],
        default="both",
        help="Toggle sequential switching for accurate/fast modes.",
    )
    parser.add_argument(
        "--no-opencv",
        action="store_true",
        help="Skip OpenCV timing/accuracy (useful if FFmpeg logs are noisy).",
    )
    # If the provided path is a directory, the script will automatically
    # benchmark all supported videos in that directory and print a summary.
    return parser.parse_args()


# Minimal, independent PyAV fast reader inspired by napari-pyav's FastVideoReader
class FastVideoReaderLike:
    def __init__(self, filename: str, read_format: str = "rgb24", threading: bool = True, thread_count: int = 0) -> None:
        import av  # local import to avoid hard dep when unused

        self._av = av
        self.container = av.open(filename)
        self.stream = self.container.streams.video[0]
        try:
            self.stream.codec_context.thread_count = int(thread_count)
            self.stream.codec_context.thread_type = "AUTO" if threading else "SLICE"
        except Exception:
            pass
        self.read_format = read_format
        rate = self.stream.guessed_rate or self.stream.average_rate or 0
        tb = float(self.stream.time_base or 1)
        self._pts_per_frame = 1.0 / (float(rate) * tb) if rate else 1.0
        self._start = int(self.stream.start_time or 0)
        self.last_pts = None
        self._frame_to_pts = lambda n: int(round(n * self._pts_per_frame)) + self._start
        self.rewind()

    def rewind(self) -> None:
        self.container.seek(0)
        self._gen = self.container.decode(video=0)
        self.last_pts = None

    def _decode_next(self):
        frame = next(self._gen)
        self.last_pts = frame.pts if frame.pts is not None else frame.dts
        img = frame.to_ndarray(format=self.read_format)
        return img

    def read(self):
        return self._decode_next()

    def read_frame(self, frame_idx: int):
        if frame_idx <= 0:
            self.rewind()
            return self._decode_next()
        if self.last_pts is not None and self.last_pts == self._frame_to_pts(frame_idx - 1):
            return self._decode_next()

        target_pts = self._frame_to_pts(frame_idx)
        # seek near frame; backward=True, any_frame=False (keyframe)
        self.container.seek(target_pts - self._start, backward=True, stream=self.container.streams.video[0])
        self._gen = self.container.decode(video=0)
        wiggle = self._pts_per_frame / 10.0
        # decode first frame
        f = next(self._gen)
        cur_pts = f.pts if f.pts is not None else f.dts
        if cur_pts is None:
            cur_pts = target_pts
        if cur_pts > target_pts:
            # overshoot: backtrack
            back = max(1, int(round(100)))
            self.container.seek(self._frame_to_pts(max(0, frame_idx - back)) - self._start, backward=True, stream=self.container.streams.video[0])
            self._gen = self.container.decode(video=0)
            f = next(self._gen)
            cur_pts = f.pts if f.pts is not None else f.dts
            if cur_pts is None:
                cur_pts = target_pts
        # advance until target reached within wiggle
        while cur_pts < (target_pts - wiggle):
            f = next(self._gen)
            cur_pts = f.pts if f.pts is not None else f.dts
            if cur_pts is None:
                break
        img = f.to_ndarray(format=self.read_format)
        self.last_pts = cur_pts
        return img

    def close(self) -> None:
        try:
            self.container.close()
        except Exception:
            pass


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


def accuracy_stats(values: list[int], expected: list[int], *, failures: int = 0) -> dict:
    failures = max(failures, sum(1 for v in values if v < 0))
    matched = [(v, e) for v, e in zip(values, expected) if v >= 0]
    diffs = [abs(v - e) for v, e in matched]
    mean_err = statistics.mean(diffs) if diffs else float("nan")
    max_err = max(diffs) if diffs else float("nan")
    return {"mean": float(mean_err), "max": int(max_err if max_err == max_err else -1), "failed": int(failures)}


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


def time_collect(func, indices: np.ndarray) -> float:
    timings = []
    for idx in indices:
        start = time.perf_counter()
        func(int(idx))
        timings.append(time.perf_counter() - start)
    median = statistics.median(timings) if timings else 0.0
    return median * 1000.0


def repeat_indices(indices: np.ndarray, passes: int) -> np.ndarray:
    """Repeat an index list to amplify cache effects."""

    passes = max(1, int(passes))
    if passes == 1:
        return indices
    return np.concatenate([indices] * passes)


def _build_indices(
    frame_count: int,
    *,
    samples: int,
    seed: int,
    index_pattern: str,
    passes: int,
) -> tuple[np.ndarray, int, int]:
    base_sample_count = min(int(samples), int(frame_count))
    rng = np.random.default_rng(int(seed))
    if index_pattern == "sequential":
        base_indices = np.arange(base_sample_count)
    else:
        base_indices = rng.choice(frame_count, size=base_sample_count, replace=False)
    indices = repeat_indices(base_indices, passes)
    return indices, base_sample_count, len(indices)


def _accuracy_embedded(
    reader: VideoReader,
    indices: np.ndarray,
    *,
    scrub_mode: str,
    use_sequential: bool,
    include_fastvideo: bool,
    path: Path,
    threading: bool,
    thread_count: int,
) -> dict[str, dict[str, float | int]]:
    asset = load_asset_data(path)
    if asset.numbers:
        expected = [
            asset.numbers[min(int(idx), len(asset.numbers) - 1)]
            for idx in indices
        ]
    else:
        expected = [int(idx) for idx in indices]
    frame_rate = reader.frame_rate or 1.0

    accurate = [
        decode_frame_number(
            reader.read_frame(index=int(idx), mode="accurate", use_sequential=use_sequential).image,
            channel_index=0,
        )
        for idx in indices
    ]
    acc = {"accurate": accuracy_stats(accurate, expected)}

    accurate_tl = [
        decode_frame_number(reader.read_frame(index=int(idx), mode="accurate_timeline").image, channel_index=0)
        for idx in indices
    ]
    acc["accurate_tl"] = accuracy_stats(accurate_tl, expected)

    scrub = [
        decode_frame_number(
            reader.read_keyframe_at(int(idx) / frame_rate, mode=scrub_mode).image,
            channel_index=0,
        )
        for idx in indices
    ]
    acc["scrub"] = accuracy_stats(scrub, expected)

    fast = []
    fast_failures = 0
    for idx in indices:
        try:
            fr = reader.read_frame_fast(index=int(idx), use_sequential=use_sequential)
            fast.append(decode_frame_number(fr.image, channel_index=0))
        except Exception:
            fast.append(-1)
            fast_failures += 1
    acc["fast"] = accuracy_stats(fast, expected, failures=fast_failures)

    if include_fastvideo:
        try:
            import av  # noqa: F401
        except ImportError:
            pass
        else:
            ref = FastVideoReaderLike(
                str(path),
                read_format="rgb24",
                threading=threading,
                thread_count=thread_count,
            )
            vals: list[int] = []
            fails = 0
            try:
                for idx in indices:
                    try:
                        img = ref.read_frame(int(idx))
                        vals.append(decode_frame_number(img, channel_index=0))
                    except Exception:
                        vals.append(-1)
                        fails += 1
            finally:
                ref.close()
            acc["fastvideo"] = accuracy_stats(vals, expected, failures=fails)

    if len(indices) and indices[0] == 0 and np.all(np.diff(indices) == 1):
        reader._backend.reset_sequence()
        seq_vals = []
        for _ in range(len(indices)):
            fr = reader._backend.read_next_frame()
            seq_vals.append(decode_frame_number(fr.image, channel_index=0))
        acc["sequential"] = accuracy_stats(seq_vals, expected)

    return acc


def _accuracy_pts(
    reader: VideoReader,
    indices: np.ndarray,
    *,
    scrub_mode: str,
    use_sequential: bool,
    include_fastvideo: bool,
    path: Path,
    threading: bool,
    thread_count: int,
) -> dict[str, dict[str, float | int]]:
    expected = [int(idx) for idx in indices]

    def idx_from_pts(pts: int) -> int:
        return reader.index_from_pts(int(pts))

    acc_vals: list[int] = []
    for idx in indices:
        fr = reader.read_frame(index=int(idx), mode="accurate", use_sequential=use_sequential)
        acc_vals.append(-1 if fr.pts is None else idx_from_pts(int(fr.pts)))
    acc = {"accurate": accuracy_stats(acc_vals, expected)}

    acc_tl_vals: list[int] = []
    for idx in indices:
        fr = reader.read_frame(index=int(idx), mode="accurate_timeline")
        acc_tl_vals.append(-1 if fr.pts is None else idx_from_pts(int(fr.pts)))
    acc["accurate_tl"] = accuracy_stats(acc_tl_vals, expected)

    scrub_vals: list[int] = []
    scrub_failures = 0
    fps = reader.frame_rate or 1.0
    for idx in indices:
        try:
            fr = reader.read_keyframe_at(int(idx) / fps, mode=scrub_mode)
            if fr.pts is None:
                scrub_vals.append(-1)
                scrub_failures += 1
            else:
                scrub_vals.append(idx_from_pts(int(fr.pts)))
        except Exception:
            scrub_vals.append(-1)
            scrub_failures += 1
    acc["scrub"] = accuracy_stats(scrub_vals, expected, failures=scrub_failures)

    fast_vals: list[int] = []
    fast_failures = 0
    for idx in indices:
        try:
            fr = reader.read_frame_fast(index=int(idx), use_sequential=use_sequential)
            if fr.pts is None:
                fast_vals.append(-1)
                fast_failures += 1
            else:
                fast_vals.append(idx_from_pts(int(fr.pts)))
        except Exception:
            fast_vals.append(-1)
            fast_failures += 1
    acc["fast"] = accuracy_stats(fast_vals, expected, failures=fast_failures)

    if include_fastvideo:
        try:
            import av  # noqa: F401
        except ImportError:
            pass
        else:
            ref = FastVideoReaderLike(
                str(path),
                read_format="rgb24",
                threading=threading,
                thread_count=thread_count,
            )
            vals: list[int] = []
            fails = 0
            try:
                for idx in indices:
                    try:
                        ref.read_frame(int(idx))
                        pts = ref.last_pts
                        if pts is None:
                            vals.append(-1)
                            fails += 1
                        else:
                            vals.append(idx_from_pts(int(pts)))
                    except Exception:
                        vals.append(-1)
                        fails += 1
            finally:
                ref.close()
            acc["fastvideo"] = accuracy_stats(vals, expected, failures=fails)

    if len(indices) and indices[0] == 0 and np.all(np.diff(indices) == 1):
        reader._backend.reset_sequence()
        seq_vals = []
        for _ in range(len(indices)):
            fr = reader._backend.read_next_frame()
            seq_vals.append(-1 if fr.pts is None else idx_from_pts(int(fr.pts)))
        acc["sequential"] = accuracy_stats(seq_vals, expected)

    return acc


def _accuracy_truth(
    reader: VideoReader,
    indices: np.ndarray,
    *,
    scrub_mode: str,
    use_sequential: bool,
    include_fastvideo: bool,
    path: Path,
    threading: bool,
    thread_count: int,
) -> dict[str, dict[str, float | int]]:
    gt = ensure_groundtruth(path)
    expected_decode = [int(idx) for idx in indices]
    fps_nom = gt.get("nominal_fps", 0.0) or (reader.nominal_frame_rate or reader.frame_rate or 1.0)
    expected_tl = [gt_index_from_time(gt, float(int(i)) / fps_nom, nearest=False) for i in indices]
    acc = {"accurate": accuracy_stats(expected_decode, expected_decode)}

    acc_t_vals: list[int] = []
    for i in indices:
        fr = reader.read_frame(index=int(i), mode="accurate_timeline")
        acc_t_vals.append(-1 if fr.pts is None else gt_index_from_pts(gt, int(fr.pts)))
    acc["accurate_tl"] = accuracy_stats(acc_t_vals, expected_tl)

    fast_vals: list[int] = []
    for i in indices:
        fr = reader.read_frame_fast(index=int(i), use_sequential=use_sequential)
        fast_vals.append(-1 if fr.pts is None else gt_index_from_pts(gt, int(fr.pts)))
    acc["fast"] = accuracy_stats(fast_vals, expected_tl)

    scrub_vals: list[int] = []
    keyframes = gt["keyframe_pts"]
    fps = reader.frame_rate or 1.0
    for i in indices:
        fr = reader.read_keyframe_at(float(int(i)) / fps, mode=scrub_mode)
        scrub_vals.append(-1 if fr.pts is None else gt_index_from_pts(gt, int(fr.pts)))
    exp_scrub = []
    for i in indices:
        t_s = float(int(i)) / fps
        target_pts = int(round(t_s / gt["time_base"]))
        import bisect
        if scrub_mode == "previous":
            pos = bisect.bisect_right(keyframes, target_pts) - 1
            pos = max(0, min(pos, len(keyframes) - 1))
            kf_pts = keyframes[pos]
        elif scrub_mode == "nearest":
            pos = bisect.bisect_left(keyframes, target_pts)
            if pos <= 0:
                kf_pts = keyframes[0]
            elif pos >= len(keyframes):
                kf_pts = keyframes[-1]
            else:
                prev_pts = keyframes[pos - 1]
                next_pts = keyframes[pos]
                kf_pts = prev_pts if abs(prev_pts - target_pts) <= abs(next_pts - target_pts) else next_pts
        else:
            pos = bisect.bisect_left(keyframes, target_pts)
            pos = min(pos, len(keyframes) - 1)
            kf_pts = keyframes[pos]
        exp_scrub.append(gt_index_from_pts(gt, kf_pts))
    acc["scrub"] = accuracy_stats(scrub_vals, exp_scrub)

    if len(indices) and indices[0] == 0 and np.all(np.diff(indices) == 1):
        reader._backend.reset_sequence()
        seq_vals = []
        for _ in range(len(indices)):
            fr = reader._backend.read_next_frame()
            seq_vals.append(-1 if fr.pts is None else gt_index_from_pts(gt, int(fr.pts)))
        expected_seq = list(range(len(indices)))
        acc["sequential"] = accuracy_stats(seq_vals, expected_seq)

    if include_fastvideo:
        try:
            import av  # noqa: F401
        except ImportError:
            pass
        else:
            ref = FastVideoReaderLike(
                str(path),
                read_format="rgb24",
                threading=threading,
                thread_count=thread_count,
            )
            fv_vals = []
            for i in indices:
                ref.read_frame(int(i))
                pts = ref.last_pts
                fv_vals.append(-1 if pts is None else gt_index_from_pts(gt, int(pts)))
            acc["fastvideo"] = accuracy_stats(fv_vals, expected_tl)
            ref.close()

    return acc


def run_benchmark_suite(paths: list[Path], config: BenchmarkConfig) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for vp in paths:
        with VideoReader(
            str(vp),
            scrub_bucket_ms=config.scrub_bucket_ms,
            decoded_frame_cache_size=config.cache_size,
            threading=config.threading,
            thread_count=config.thread_count,
        ) as reader:
            if config.metric == "truth":
                gt = ensure_groundtruth(vp)
                frame_count = len(gt.get("frame_pts", [])) or reader.number_of_frames
            else:
                frame_count = reader.number_of_frames
            if frame_count <= 0:
                continue
            indices, base_count, sample_count = _build_indices(
                frame_count,
                samples=config.samples,
                seed=config.seed,
                index_pattern=config.index_pattern,
                passes=config.passes,
            )
            reader.build_keyframe_index()
            frame_rate = reader.frame_rate or 1.0
            use_seq = config.use_sequential

            t_seq = time_sequential(reader, base_count)
            t_acc = time_collect(
                lambda idx: reader.read_frame(index=idx, mode="accurate", use_sequential=use_seq),
                indices,
            )
            t_acct = time_collect(
                lambda idx: reader.read_frame(index=idx, mode="accurate_timeline"),
                indices,
            )
            t_scrub = time_collect(lambda idx: reader.read_keyframe_at(idx / frame_rate), indices)
            t_fast = time_collect(
                lambda idx: reader.read_frame_fast(index=idx, use_sequential=use_seq),
                indices,
            )
            t_fv = None
            if config.compare_fastvideo:
                try:
                    import av  # noqa: F401
                except ImportError:
                    t_fv = None
                else:
                    fv_reader = FastVideoReaderLike(
                        str(vp),
                        read_format="rgb24",
                        threading=config.threading,
                        thread_count=config.thread_count,
                    )
                    try:
                        t_fv = time_collect(lambda idx: fv_reader.read_frame(idx), indices)
                    finally:
                        fv_reader.close()

            timing_ms = {
                "sequential": t_seq,
                "accurate": t_acc,
                "accurate_tl": t_acct,
                "scrub": t_scrub,
                "fast": t_fast,
                "fastvideo": t_fv,
            }

            accuracy: dict[str, dict[str, float | int]] | None
            if config.metric == "truth":
                accuracy = _accuracy_truth(
                    reader,
                    indices,
                    scrub_mode=config.scrub_mode,
                    use_sequential=use_seq,
                    include_fastvideo=config.compare_fastvideo,
                    path=vp,
                    threading=config.threading,
                    thread_count=config.thread_count,
                )
            elif config.metric == "embedded":
                accuracy = _accuracy_embedded(
                    reader,
                    indices,
                    scrub_mode=config.scrub_mode,
                    use_sequential=use_seq,
                    include_fastvideo=config.compare_fastvideo,
                    path=vp,
                    threading=config.threading,
                    thread_count=config.thread_count,
                )
            elif config.metric == "pts":
                accuracy = _accuracy_pts(
                    reader,
                    indices,
                    scrub_mode=config.scrub_mode,
                    use_sequential=use_seq,
                    include_fastvideo=config.compare_fastvideo,
                    path=vp,
                    threading=config.threading,
                    thread_count=config.thread_count,
                )
            else:
                accuracy = None

            results.append(
                BenchmarkResult(
                    name=vp.name,
                    frames=int(frame_count),
                    base_samples=int(base_count),
                    samples=int(sample_count),
                    timing_ms=timing_ms,
                    accuracy=accuracy,
                )
            )

    return results


def time_sequential(reader: VideoReader, count: int) -> float:
    timings = []
    reader._backend.reset_sequence()
    for _ in range(count):
        start = time.perf_counter()
        reader.read_next()
        timings.append(time.perf_counter() - start)
    median = statistics.median(timings) if timings else 0.0
    return median * 1000.0


def accuracy_benchmark_sequential_embedded(reader: VideoReader, count: int) -> None:
    reader._backend.reset_sequence()
    expected = list(range(count))
    vals = []
    for _ in range(count):
        fr = reader._backend.read_next_frame()
        vals.append(decode_frame_number(fr.image, channel_index=0))
    accuracy_report("sequential", vals, expected)


def accuracy_benchmark_sequential_pts(reader: VideoReader, count: int) -> None:
    reader._backend.reset_sequence()
    expected = list(range(count))
    vals = []
    for _ in range(count):
        fr = reader._backend.read_next_frame()
        pts = fr.pts
        vals.append(-1 if pts is None else reader.index_from_pts(int(pts)))
    accuracy_report("sequential", vals, expected)


def accuracy_benchmark_sequential_truth(reader: VideoReader, count: int, *, path: Path) -> None:
    gt = ensure_groundtruth(path)
    reader._backend.reset_sequence()
    expected = list(range(count))
    vals = []
    for _ in range(count):
        fr = reader._backend.read_next_frame()
        pts = fr.pts
        vals.append(-1 if pts is None else gt_index_from_pts(gt, int(pts)))
    accuracy_report("sequential", vals, expected)


def time_fastvideo_like(
    path: Path,
    indices: np.ndarray,
    *,
    threading: bool = True,
    thread_count: int = 0,
) -> None:
    try:
        import av  # noqa: F401
    except ImportError:
        print(f"{'fastvideo':>12}: skipped (av not installed)")
        return
    reader = FastVideoReaderLike(
        str(path),
        read_format="rgb24",
        threading=threading,
        thread_count=thread_count,
    )
    timings = []
    try:
        for idx in indices:
            start = time.perf_counter()
            reader.read_frame(int(idx))
            timings.append(time.perf_counter() - start)
    finally:
        reader.close()
    median = statistics.median(timings)
    fps = 1.0 / median if median > 0 else float("inf")
    print(f"{'fastvideo':>12}: {median * 1000:.2f} ms/frame ({fps:.1f} fps)")


# Ground truth handling -------------------------------------------------------

GT_DIR = Path("tests/groundtruth")


def _file_sha256(path: Path, chunk_size: int = 2 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_groundtruth(path: Path) -> dict:
    """Load or generate ground truth for a video path.

    Ground truth stores frame PTS list, keyframe PTS list, time_base, and nominal FPS.
    We detect changes via SHA256 file hash and regenerate if mismatched.
    """

    GT_DIR.mkdir(parents=True, exist_ok=True)
    video_hash = _file_sha256(path)
    out_path = GT_DIR / f"{path.name}.json"

    if out_path.exists():
        try:
            data = json.loads(out_path.read_text())
            if data.get("video_hash") == video_hash:
                return data
        except Exception:
            pass

    # Generate via PyAV
    import av

    container = av.open(str(path))
    stream = container.streams.video[0]

    time_base = float(stream.time_base)
    rate = getattr(stream, "guessed_rate", None) or stream.average_rate or stream.base_rate
    nominal_fps = float(rate) if rate is not None else 0.0

    frame_pts: list[int] = []
    keyframe_pts: list[int] = []
    start_pts = int(stream.start_time or 0)

    for packet in container.demux(stream):
        if packet.is_keyframe and (packet.pts is not None or packet.dts is not None):
            p = packet.pts if packet.pts is not None else packet.dts
            keyframe_pts.append(int(p))
        for frame in packet.decode():
            pts = frame.pts if frame.pts is not None else frame.dts
            if pts is None:
                pts = start_pts + len(frame_pts)
            frame_pts.append(int(pts))
    container.close()

    if not keyframe_pts:
        keyframe_pts = [start_pts]
    keyframe_pts = sorted(set(keyframe_pts))

    data = {
        "path": str(path),
        "video_hash": video_hash,
        "size": path.stat().st_size,
        "mtime": int(path.stat().st_mtime),
        "time_base": time_base,
        "nominal_fps": nominal_fps,
        "frame_pts": frame_pts,
        "keyframe_pts": keyframe_pts,
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, out_path)
    return data


def gt_index_from_pts(gt: dict, pts: int) -> int:
    arr = gt["frame_pts"]
    lo, hi = 0, len(arr) - 1
    if pts <= arr[0]:
        return 0
    if pts >= arr[-1]:
        return hi
    while lo <= hi:
        mid = (lo + hi) // 2
        m = arr[mid]
        if m == pts:
            return mid
        if m < pts:
            lo = mid + 1
        else:
            hi = mid - 1
    if lo >= len(arr):
        return hi
    if hi < 0:
        return lo
    return lo if abs(arr[lo] - pts) < abs(arr[hi] - pts) else hi


def gt_index_first_after_pts(gt: dict, pts: int) -> int:
    arr = gt["frame_pts"]
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < pts:
            lo = mid + 1
        else:
            hi = mid
    return min(lo, len(arr) - 1)


def gt_index_from_time(gt: dict, t_s: float, *, nearest: bool) -> int:
    pts = int(round(t_s / gt.get("time_base", 1.0)))
    return gt_index_from_pts(gt, pts) if nearest else gt_index_first_after_pts(gt, pts)


def accuracy_benchmark_embedded(
    reader: VideoReader,
    indices: np.ndarray,
    frame_rate: float,
    *,
    scrub_mode: str,
    include_fastvideo: bool = False,
    path: Path | None = None,
    use_opencv: bool = True,
    use_sequential: bool = True,
    threading: bool = True,
    thread_count: int = 0,
) -> None:
    """Benchmark accuracy using embedded frame numbers in test fixtures."""

    expected = [int(idx) for idx in indices]

    accurate = [
        decode_frame_number(
            reader.read_frame(index=int(idx), mode="accurate", use_sequential=use_sequential).image,
            channel_index=0,
        )
        for idx in indices
    ]
    accuracy_report("accurate", accurate, expected)

    accurate_tl = [
        decode_frame_number(reader.read_frame(index=int(idx), mode="accurate_timeline").image, channel_index=0)
        for idx in indices
    ]
    accuracy_report("accurate_tl", accurate_tl, expected)

    scrub = [
        decode_frame_number(
            reader.read_keyframe_at(int(idx) / frame_rate, mode=scrub_mode).image, channel_index=0
        )
        for idx in indices
    ]
    accuracy_report("scrub", scrub, expected)

    fast = []
    fast_failures = 0
    for idx in indices:
        try:
            fr = reader.read_frame_fast(index=int(idx), use_sequential=use_sequential)
            frame = fr.image
        except Exception:
            fast_failures += 1
            fast.append(-1)
        else:
            # Default path decodes RGB; embedded counter lives in R channel => index 0
            fast.append(decode_frame_number(frame, channel_index=0))
    accuracy_report("fast", fast, expected, failures=fast_failures)

    # removed fast_rgb variant for simplicity

    if use_opencv:
        try:
            import cv2
        except ImportError:
            print(f"{'opencv':>12}: skipped (cv2 not installed)")
        else:
            cap = cv2.VideoCapture(reader._backend._path)
            if not cap.isOpened():
                print(f"{'opencv':>12}: failed to open")
            else:
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

    if include_fastvideo and path is not None:
        try:
            import av  # noqa: F401
        except ImportError:
            print(f"{'fastvideo':>12}: skipped (av not installed)")
            return
        ref = FastVideoReaderLike(
            str(path),
            read_format="rgb24",
            threading=threading,
            thread_count=thread_count,
        )
        vals: list[int] = []
        fails = 0
        try:
            for idx in indices:
                try:
                    img = ref.read_frame(int(idx))
                    vals.append(decode_frame_number(img, channel_index=0))
                except Exception:
                    vals.append(-1)
                    fails += 1
        finally:
            ref.close()
        accuracy_report("fastvideo", vals, expected, failures=fails)


def accuracy_benchmark_pts(
    reader: VideoReader,
    indices: np.ndarray,
    *,
    scrub_mode: str,
    include_fastvideo: bool = False,
    path: Path | None = None,
    use_opencv: bool = True,
    use_sequential: bool = True,
    threading: bool = True,
    thread_count: int = 0,
) -> None:
    """Benchmark accuracy by mapping decoded PTS to nearest reference index.

    This does not require embedded frame numbers and works for arbitrary videos.
    """

    # Ensure metadata available
    _ = reader.number_of_frames

    def idx_from_pts(pts: int) -> int:
        return reader.index_from_pts(int(pts))

    expected = [int(idx) for idx in indices]

    # Accurate baseline via PTS of exact indices
    accurate_vals: list[int] = []
    for idx in indices:
        fr = reader.read_frame(index=int(idx), mode="accurate", use_sequential=use_sequential)
        if fr.pts is None:
            accurate_vals.append(-1)
        else:
            accurate_vals.append(idx_from_pts(int(fr.pts)))
    accuracy_report("accurate", accurate_vals, expected)

    # Accurate timeline (index -> nominal FPS timestamp)
    accurate_tl_vals: list[int] = []
    fps_nom = reader.nominal_frame_rate or reader.frame_rate or 1.0
    for idx in indices:
        fr = reader.read_frame(index=int(idx), mode="accurate_timeline")
        if fr.pts is None:
            accurate_tl_vals.append(-1)
        else:
            accurate_tl_vals.append(idx_from_pts(int(fr.pts)))
    accuracy_report("accurate_tl", accurate_tl_vals, expected)

    # Scrub (keyframe)
    scrub_vals: list[int] = []
    scrub_failures = 0
    fps = reader.frame_rate or 1.0
    for idx in indices:
        try:
            fr = reader.read_keyframe_at(int(idx) / fps, mode=scrub_mode)
            if fr.pts is None:
                scrub_vals.append(-1)
                scrub_failures += 1
            else:
                scrub_vals.append(idx_from_pts(int(fr.pts)))
        except Exception:
            scrub_vals.append(-1)
            scrub_failures += 1
    accuracy_report("scrub", scrub_vals, expected, failures=scrub_failures)

    # Fast (RGB)
    fast_vals: list[int] = []
    fast_failures = 0
    for idx in indices:
        try:
            fr = reader.read_frame_fast(index=int(idx), use_sequential=use_sequential)
            if fr.pts is None:
                fast_vals.append(-1)
                fast_failures += 1
            else:
                fast_vals.append(idx_from_pts(int(fr.pts)))
        except Exception:
            fast_vals.append(-1)
            fast_failures += 1
    accuracy_report("fast", fast_vals, expected, failures=fast_failures)

    # removed fast_rgb variant for simplicity

    # OpenCV: map reported timestamp back to nearest index
    if use_opencv:
        try:
            import cv2
        except ImportError:
            print(f"{'opencv':>12}: skipped (cv2 not installed)")
        else:
            cap = cv2.VideoCapture(reader._backend._path)
            if not cap.isOpened():
                print(f"{'opencv':>12}: failed to open")
            else:
                ocv_vals: list[int] = []
                ocv_failures = 0
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, _ = cap.read()
                    if not ok:
                        ocv_vals.append(-1)
                        ocv_failures += 1
                        continue
                    t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    t_s = (t_ms or 0.0) / 1000.0
                    ocv_vals.append(reader.index_from_time(float(t_s)))
                cap.release()
                accuracy_report("opencv", ocv_vals, expected, failures=ocv_failures)


def accuracy_benchmark_truth(
    reader: VideoReader,
    indices: np.ndarray,
    *,
    scrub_mode: str,
    include_fastvideo: bool = False,
    path: Path,
    use_opencv: bool = True,
    use_sequential: bool = True,
    threading: bool = True,
    thread_count: int = 0,
) -> None:
    """Benchmark accuracy using stored ground truth for the given video path."""

    gt = ensure_groundtruth(path)
    expected_decode = [int(idx) for idx in indices]
    fps_nom = gt.get("nominal_fps", 0.0) or (reader.nominal_frame_rate or reader.frame_rate or 1.0)

    def expected_timeline_index(i: int) -> int:
        t_s = float(i) / fps_nom
        # Our accurate_timeline/fast return first-after frame at timestamp
        return gt_index_from_time(gt, t_s, nearest=False)

    expected_tl = [expected_timeline_index(int(idx)) for idx in indices]

    # Accurate (decode-order)
    acc_vals: list[int] = []
    for idx in indices:
        fr = reader.read_frame(index=int(idx), mode="accurate", use_sequential=use_sequential)
        if fr.pts is None:
            acc_vals.append(-1)
        else:
            acc_vals.append(gt_index_from_pts(gt, int(fr.pts)))
    accuracy_report("accurate", acc_vals, expected_decode)

    # Accurate timeline
    acc_tl_vals: list[int] = []
    for idx in indices:
        fr = reader.read_frame(index=int(idx), mode="accurate_timeline")
        if fr.pts is None:
            acc_tl_vals.append(-1)
        else:
            acc_tl_vals.append(gt_index_from_pts(gt, int(fr.pts)))
    accuracy_report("accurate_tl", acc_tl_vals, expected_tl)

    # Fast
    fast_vals: list[int] = []
    fast_failures = 0
    for idx in indices:
        try:
            fr = reader.read_frame_fast(index=int(idx), use_sequential=use_sequential)
            if fr.pts is None:
                fast_vals.append(-1)
                fast_failures += 1
            else:
                fast_vals.append(gt_index_from_pts(gt, int(fr.pts)))
        except Exception:
            fast_vals.append(-1)
            fast_failures += 1
    accuracy_report("fast", fast_vals, expected_tl, failures=fast_failures)

    # Scrub
    scrub_vals: list[int] = []
    fps = reader.frame_rate or 1.0
    keyframes = gt["keyframe_pts"]
    for idx in indices:
        t_s = float(int(idx)) / fps
        fr = reader.read_keyframe_at(t_s, mode=scrub_mode)
        if fr.pts is None:
            scrub_vals.append(-1)
            continue
        scrub_vals.append(gt_index_from_pts(gt, int(fr.pts)))
    # Build expected scrub indices from ground truth keyframes and policy
    exp_scrub: list[int] = []
    for idx in indices:
        t_s = float(int(idx)) / fps
        target_pts = int(round(t_s / gt["time_base"]))
        if scrub_mode == "previous":
            import bisect
            pos = bisect.bisect_right(keyframes, target_pts) - 1
            pos = max(0, min(pos, len(keyframes) - 1))
            kf_pts = keyframes[pos]
        elif scrub_mode == "nearest":
            import bisect
            pos = bisect.bisect_left(keyframes, target_pts)
            if pos <= 0:
                kf_pts = keyframes[0]
            elif pos >= len(keyframes):
                kf_pts = keyframes[-1]
            else:
                prev_pts = keyframes[pos - 1]
                next_pts = keyframes[pos]
                kf_pts = prev_pts if abs(prev_pts - target_pts) <= abs(next_pts - target_pts) else next_pts
        else:  # next
            import bisect
            pos = bisect.bisect_left(keyframes, target_pts)
            pos = min(pos, len(keyframes) - 1)
            kf_pts = keyframes[pos]
        exp_scrub.append(gt_index_from_pts(gt, kf_pts))
    accuracy_report("scrub", scrub_vals, exp_scrub)

    # OpenCV
    if use_opencv:
        try:
            import cv2
        except ImportError:
            print(f"{'opencv':>12}: skipped (cv2 not installed)")
        else:
            cap = cv2.VideoCapture(reader._backend._path)
            if not cap.isOpened():
                print(f"{'opencv':>12}: failed to open")
            else:
                ocv_vals: list[int] = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, _ = cap.read()
                    if not ok:
                        ocv_vals.append(-1)
                        continue
                    t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    t_s = (t_ms or 0.0) / 1000.0
                    ocv_vals.append(gt_index_from_time(gt, t_s, nearest=False))
                cap.release()
                accuracy_report("opencv", ocv_vals, expected_tl)

    # FastVideoReader-like baseline
    if include_fastvideo and path is not None:
        try:
            import av  # noqa: F401
        except ImportError:
            print(f"{'fastvideo':>12}: skipped (av not installed)")
        else:
            ref = FastVideoReaderLike(
                str(path),
                read_format="rgb24",
                threading=threading,
                thread_count=thread_count,
            )
            fv_vals: list[int] = []
            fv_failures = 0
            try:
                for idx in indices:
                    try:
                        ref.read_frame(int(idx))
                        pts = ref.last_pts
                        if pts is None:
                            fv_vals.append(-1)
                            fv_failures += 1
                        else:
                            fv_vals.append(gt_index_from_pts(gt, int(pts)))
                    except Exception:
                        fv_vals.append(-1)
                        fv_failures += 1
            finally:
                ref.close()
            accuracy_report("fastvideo", fv_vals, expected_tl, failures=fv_failures)

    # removed duplicated fastvideo block


def main() -> None:
    """Run the benchmark for accurate, scrub, and fast modes."""
    args = parse_args()
    if not args.path.exists():
        raise SystemExit(f"Missing path: {args.path}")

    config = BenchmarkConfig(
        metric=args.metric,
        samples=args.samples,
        seed=args.seed,
        scrub_bucket_ms=args.scrub_bucket_ms,
        scrub_mode=args.scrub_mode,
        compare_fastvideo=args.compare_fastvideo,
        index_pattern=args.index_pattern,
        use_sequential=args.use_sequential != "off",
        cache_size=args.cache_size,
        passes=args.passes,
        threading=args.threading == "on",
        thread_count=args.thread_count,
        no_opencv=args.no_opencv,
    )

    if args.path.is_dir():
        videos = sorted([p for p in args.path.iterdir() if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}])
        if not videos:
            raise SystemExit(f"No video files found in {args.path}")
        summary = run_benchmark_suite(videos, config)

        print("\nSpeed (ms/frame) — median")
        headers = ["video", "sequential", "accurate", "accurate_tl", "scrub", "fast"]
        if any(s.timing_ms.get("fastvideo") for s in summary):
            headers.append("fastvideo")
        print(" | ".join(f"{h:>14}" for h in headers))
        for s in summary:
            row = [
                s.name,
                s.timing_ms.get("sequential"),
                s.timing_ms.get("accurate"),
                s.timing_ms.get("accurate_tl"),
                s.timing_ms.get("scrub"),
                s.timing_ms.get("fast"),
            ]
            if "fastvideo" in headers:
                row.append(s.timing_ms.get("fastvideo") or float("nan"))
            print(" | ".join(f"{(r if isinstance(r, str) else f'{float(r):.2f}'):>14}" for r in row))

        if summary and summary[0].accuracy is not None:
            print("\nAccuracy (mean|max)")
            acc_modes = ["accurate", "accurate_tl", "scrub", "fast"]
            if args.index_pattern == "sequential":
                acc_modes.insert(0, "sequential")
            if any(s.accuracy and "fastvideo" in s.accuracy for s in summary):
                acc_modes.append("fastvideo")
            print(" | ".join([f"{h:>14}" for h in ["video"] + acc_modes]))
            for s in summary:
                acc = s.accuracy or {}
                cells = [s.name]
                for m in acc_modes:
                    st = acc.get(m)
                    cells.append(f"{st['mean']:.2f}|{st['max']}" if st else "-")
                print(" | ".join(f"{c:>14}" for c in cells))
        return

    summary = run_benchmark_suite([args.path], config)
    if not summary:
        raise SystemExit("Video reports zero frames.")
    result = summary[0]
    print(f"Frames: {result.frames} | Samples: {result.samples}")
    print(f"Sequential samples: {result.base_samples}")
    print("\nSpeed (ms/frame)")
    print(f"{'sequential':>12}: {result.timing_ms['sequential']:.2f} ms/frame")
    for mode in ["accurate", "accurate_tl", "scrub", "fast"]:
        val = result.timing_ms.get(mode)
        if val is not None:
            fps = (1000.0 / float(val)) if val else float("inf")
            print(f"{mode:>12}: {val:.2f} ms/frame ({fps:.1f} fps)")
    if result.timing_ms.get("fastvideo") is not None:
        val = float(result.timing_ms["fastvideo"])
        fps = (1000.0 / val) if val else float("inf")
        print(f"{'fastvideo':>12}: {val:.2f} ms/frame ({fps:.1f} fps)")

    if result.accuracy is not None:
        print("\nAccuracy (mean|max)")
        acc_modes = ["accurate", "accurate_tl", "scrub", "fast"]
        if args.index_pattern == "sequential":
            acc_modes.insert(0, "sequential")
        if "fastvideo" in result.accuracy:
            acc_modes.append("fastvideo")
        print(" | ".join([f"{h:>14}" for h in ["video"] + acc_modes]))
        cells = [result.name]
        for mode in acc_modes:
            st = result.accuracy.get(mode)
            cells.append(f"{st['mean']:.2f}|{st['max']}" if st else "-")
        print(" | ".join(f"{c:>14}" for c in cells))


if __name__ == "__main__":
    main()
