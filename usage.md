# Usage

## Basic access
```python
from acvr import VideoReader

with VideoReader("/path/to/video.mp4") as reader:
    frame = reader[100]
    print(reader.frame_rate)
```

## Iteration
```python
from acvr import VideoReader

reader = VideoReader("/path/to/video.mp4")
for frame in reader:
    # process frame
    pass
reader.close()
```

Iteration uses the sequential decoder internally, which is much faster than
per-frame seeks when you are processing the video in order.

## Sequential reads
Sequential reads are the fastest path for in-order decoding. Restart the
sequential decoder by creating a new reader.
```python
from acvr import VideoReader

reader = VideoReader("/path/to/video.mp4")

frame0 = reader.read_next()
frame1 = reader.read_next()

reader.close()
```

Sequential reads decode the next frame in the stream without seeking. Use them
for dense, in-order processing pipelines (e.g., feature extraction, encoding).

## Accurate timestamp reads
```python
from acvr import VideoReader

reader = VideoReader("/path/to/video.mp4")
frame = reader.read_frame_at(1.25)
reader.close()
```

## Fast scrubbing
Keyframe reads default to the nearest keyframe.
```python
from acvr import VideoReader

reader = VideoReader("/path/to/video.mp4")
keyframe = reader.read_keyframe_at(2.0)
reader.close()
```

## Selectable read modes
Pick the access mode depending on accuracy and latency needs.
```python
from acvr import VideoReader

reader = VideoReader("/path/to/video.mp4")

frame = reader.read_frame(index=120, mode="accurate")

frame_tl = reader.read_frame(index=120, mode="accurate_timeline")

frame = reader.read_frame(index=120, mode="fast")

frame = reader.read_frame(t_s=2.0, mode="scrub")

frame = reader.read_frame(index=121, mode="fast", use_sequential=True)

reader.close()
```

## Scrub tuning
For tighter scrub accuracy, use nearest mode (default) and a smaller bucket.
Smaller buckets improve precision but may increase memory churn.
```python
from acvr import VideoReader

reader = VideoReader("/path/to/video.mp4", scrub_bucket_ms=25)
keyframe = reader.read_keyframe_at(1.5)
preview = reader.read_frame(t_s=1.5, mode="scrub")
reader.close()
```

## Color conversion and performance
All modes decode RGB by default for consistent pixel values. To squeeze out a
small performance gain, set decode_rgb=False on read_frame/read_frame_fast/
read_keyframe_at to avoid RGB conversion. This typically improves speed
marginally and returns BGR for fast paths.
```python
from acvr import VideoReader
reader = VideoReader("/path/to/video.mp4")
frame = reader.read_frame(index=120, mode="fast", decode_rgb=False)
reader.close()
```

## Global indexing policy
If your application thinks in timeline frames, [] indexing can use the nominal
timeline mapping globally.
```python
from acvr import VideoReader

reader = VideoReader("/path/to/video.mp4", index_policy="timeline")
frame_100 = reader[100]
reader.close()
```

## Indexing semantics: decode-order vs timeline
- accurate: index refers to decode-order frame number (exact on CFR; may differ from nominal
  timeline on VFR). Use when you need deterministic decode positions.
- accurate_timeline: index is mapped to timestamp using nominal FPS (guessed_rate), then the
  nearest frame at/after that timestamp is returned (better alignment on VFR assets).
- fast: approximate, timeline-oriented path with very low latency; aligns well with nominal
  timeline on VFR.
- scrub: returns keyframes only; very fast and approximate for preview.

## Sequential switching
`read_frame` and `read_frame_fast` default to `use_sequential=True`, which means they
switch to the sequential decoder when indices are consecutive. This preserves accuracy
while avoiding per-frame seeks in ordered workloads. Disable it if you are mixing random
and sequential access and want consistent seek behavior.

See also: the "Indexing & Modes" page for a fuller discussion and recommendations.

## Choosing a read mode
- Sequential (`read_next` or iteration): fastest for contiguous reads; no seeking.
- Accurate: exact decode-order frame access; use for deterministic per-frame analysis.
- Accurate timeline: aligns indices to nominal FPS timestamps; preferred for VFR timeline analysis.
- Fast: low-latency approximation; good for interactive previews.
- Scrub: keyframes only; best for rapid skimming/thumbnail previews.

## Benchmarking
See the Benchmarking page for the full benchmark suite and reproducible runs:
https://janclemenslab.org/acvr/benchmark/.

## Index build cost and random seeks
`build_index` (default True) performs a full packet scan to build a keyframe index up front.
That scan cost scales with video duration/bitrate, but it lets accurate modes
seek from the nearest keyframe instead of from the requested PTS. That reduces
decode work per random seek in `accurate` (index-based) and `accurate_timeline`
modes, and it is required for `scrub` reads.

Use the benchmark script below to quantify the tradeoff on your assets:
```bash
python scripts/benchmark_index_build.py --samples 40 --max-frame-index 3000
```

It prints a Markdown table with per-video init time (with/without index), index
build cost, and per-mode median read latency. Results are machine- and storage-
dependent; regenerate them after hardware or codec changes.

Benchmark (Mac M1, sample=40, max-frame-index=3000):
| Video | Build index | Init (ms) | Build index (ms) | Accurate (ms/frame) | Accurate_tl (ms/frame) | Fast (ms/frame) | Scrub (ms/frame) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10-03-22_Test 25_1-2v1-3.mp4 | no | 21.78 | n/a | 21.93 | 28.00 | 5.99 | n/a |
| 10-03-22_Test 25_1-2v1-3.mp4 | yes | 149.02 | 124.15 | 22.00 | 29.23 | 5.95 | 8.78 |
| 20190128_113421.mp4 | no | 19.52 | n/a | 37.96 | 38.07 | 17.59 | n/a |
| 20190128_113421.mp4 | yes | 86.55 | 77.03 | 38.00 | 38.46 | 18.06 | 40.53 |
| localhost-20181120_144618.mp4 | no | 40.06 | n/a | 151.87 | 157.86 | 88.62 | n/a |
| localhost-20181120_144618.mp4 | yes | 286.39 | 176.90 | 135.79 | 138.40 | 84.34 | 89.11 |
| rpi9-20210409_093149.mp4 | no | 5.21 | n/a | 108.27 | 105.72 | 29.59 | n/a |
| rpi9-20210409_093149.mp4 | yes | 203.04 | 64.61 | 100.80 | 105.79 | 30.31 | 23.68 |
| test_cfr_h264.mp4 | no | 3.08 | n/a | 46.78 | 49.50 | 14.24 | n/a |
| test_cfr_h264.mp4 | yes | 24.08 | 19.01 | 42.60 | 43.40 | 14.99 | 9.49 |
| test_noindex_h264.mp4 | no | 9.04 | n/a | 42.95 | 46.57 | 13.94 | n/a |
| test_noindex_h264.mp4 | yes | 22.41 | 19.26 | 46.74 | 45.87 | 13.93 | 9.74 |
| test_vfr_h264.mp4 | no | 3.19 | n/a | 46.77 | 49.32 | 14.92 | n/a |
| test_vfr_h264.mp4 | yes | 21.84 | 18.17 | 43.71 | 48.36 | 17.26 | 11.04 |

Decision checklist:
- Enable `build_index=True` when you need `scrub` reads or you will issue many random
  `accurate`/`accurate_timeline` seeks on longer clips.
- Keep `build_index=False` for one-off reads, short clips, or when startup latency matters
  more than per-seek speed.
