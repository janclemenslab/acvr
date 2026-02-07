"""Benchmark keyframe index build overhead and read timings."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from acvr import VideoReader


MODES_WITHOUT_INDEX = ("accurate", "accurate_timeline", "fast")
MODES_WITH_INDEX = ("accurate", "accurate_timeline", "fast", "scrub")


def _sample_indices(
    frame_count: int,
    *,
    samples: int,
    seed: int,
    max_frame_index: int | None,
) -> np.ndarray:
    total = max(1, int(frame_count))
    if max_frame_index is not None:
        total = min(total, max(1, int(max_frame_index)))
    sample_count = min(int(samples), total)
    rng = np.random.default_rng(int(seed))
    if sample_count == total:
        return np.arange(sample_count)
    return rng.choice(total, size=sample_count, replace=False)


def _time_init(path: Path, *, build_index: bool) -> float:
    start = time.perf_counter()
    reader = VideoReader(str(path), build_index=build_index)
    init_s = time.perf_counter() - start
    reader.close()
    return init_s * 1000.0


def _time_mode(reader: VideoReader, mode: str, indices: np.ndarray) -> float:
    timings = []
    for idx in indices:
        start = time.perf_counter()
        reader.read_frame(index=int(idx), mode=mode)
        timings.append(time.perf_counter() - start)
    median_s = statistics.median(timings) if timings else 0.0
    return median_s * 1000.0


def _apply_groundtruth(reader: VideoReader, path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    gt_path = root / "tests" / "groundtruth" / f"{path.name}.json"
    if not gt_path.exists():
        return
    data = json.loads(gt_path.read_text())
    frame_pts = data.get("frame_pts")
    if isinstance(frame_pts, list) and frame_pts:
        reader._backend._frame_pts = [int(p) for p in frame_pts]
        reader._backend._frame_count = len(frame_pts)


def benchmark_paths(
    paths: list[Path],
    *,
    samples: int,
    seed: int,
    max_frame_index: int | None,
    max_packets: int | None,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for path in paths:
        init_no = _time_init(path, build_index=False)
        init_yes = _time_init(path, build_index=True)

        with VideoReader(str(path), build_index=False) as reader:
            _apply_groundtruth(reader, path)
            indices = _sample_indices(
                reader.number_of_frames,
                samples=samples,
                seed=seed,
                max_frame_index=max_frame_index,
            )
            timings_no = {mode: _time_mode(reader, mode, indices) for mode in MODES_WITHOUT_INDEX}
            build_start = time.perf_counter()
            reader.build_keyframe_index(max_packets=max_packets)
            build_ms = (time.perf_counter() - build_start) * 1000.0
            timings_yes = {mode: _time_mode(reader, mode, indices) for mode in MODES_WITH_INDEX}

        rows.append(
            {
                "video": path.name,
                "init_no": init_no,
                "init_yes": init_yes,
                "build_ms": build_ms,
                **{f"no_{k}": v for k, v in timings_no.items()},
                **{f"yes_{k}": v for k, v in timings_yes.items()},
            }
        )
    return rows


def print_markdown(rows: list[dict[str, float | str]]) -> None:
    headers = [
        "Video",
        "Build index",
        "Init (ms)",
        "Build index (ms)",
        "Accurate (ms/frame)",
        "Accurate_tl (ms/frame)",
        "Fast (ms/frame)",
        "Scrub (ms/frame)",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        video = row["video"]
        for build_index in ("no", "yes"):
            init_ms = row["init_no"] if build_index == "no" else row["init_yes"]
            build_ms = row["build_ms"] if build_index == "yes" else "n/a"
            acc = row.get(f"{build_index}_accurate", "n/a")
            acc_tl = row.get(f"{build_index}_accurate_timeline", "n/a")
            fast = row.get(f"{build_index}_fast", "n/a")
            scrub = row.get(f"{build_index}_scrub", "n/a")
            fmt = lambda v: f"{float(v):.2f}" if isinstance(v, (float, int)) else str(v)
            print(
                "| "
                + " | ".join(
                    [
                        str(video),
                        "yes" if build_index == "yes" else "no",
                        fmt(init_ms),
                        fmt(build_ms),
                        fmt(acc),
                        fmt(acc_tl),
                        fmt(fast),
                        fmt(scrub),
                    ]
                )
                + " |"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark index build vs read modes.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional list of video paths to benchmark.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory of videos to benchmark (defaults to tests/data).",
    )
    parser.add_argument("--samples", type=int, default=40, help="Number of random samples per mode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling indices.")
    parser.add_argument(
        "--max-frame-index",
        type=int,
        default=3000,
        help="Limit sampling to the first N frames (set to 0 for full range).",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=0,
        help="Cap packets scanned for the keyframe index (0 for full scan).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.paths:
        paths = [p for p in args.paths if p.exists()]
    else:
        data_dir = args.data_dir or (root / "tests" / "data")
        paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov"})
    max_frame_index = None if args.max_frame_index == 0 else args.max_frame_index
    max_packets = None if args.max_packets == 0 else args.max_packets
    rows = benchmark_paths(
        paths,
        samples=args.samples,
        seed=args.seed,
        max_frame_index=max_frame_index,
        max_packets=max_packets,
    )
    print_markdown(rows)


if __name__ == "__main__":
    main()
