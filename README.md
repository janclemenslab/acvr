# Accurate video reader (acvr)
Video reader built around PyAV for frame-accurate seeking.

Supports:
- accurate, random-access retrieval of individual frames by index or timestamp
- fast sequential reading through indexing or iterator-style access
- works with variable-frame rate videos
- accurate, fast, and scrub read modes for different latency/precision tradeoffs
- optional keyframe index for faster accurate seeks
- configurable LRU caches for decoded frames and scrub keyframe buckets to speed repeat access

Documentation at [https://janclemenslab.org/acvr]().


## Installation
In a terminal window run:
```shell
pip install acvr
```


## Usage
Open a video file and read frame 100:
```python
from acvr import VideoReader
vr = VideoReader(video_file_name)
print(vr)  # prints video_file_name, number of frames, frame rate and frame size
frame = vr[100]
vr.close()
```

Or use a context manager which takes care of opening and closing the video:
```python
with VideoReader(video_file_name) as vr:  # load the video
    frame = vr[100]
```

### Read modes
```python
from acvr import VideoReader

with VideoReader(video_file_name) as vr:
    accurate = vr.read_frame(index=100, mode="accurate")
    # VFR-aware accurate: map index -> timestamp using nominal FPS (guessed_rate)
    accurate_tl = vr.read_frame(index=100, mode="accurate_timeline")
    fast = vr.read_frame(index=100, mode="fast")
    # Scrub defaults to nearest keyframe; adjust with keyframe_mode if needed
    scrub = vr.read_frame(t_s=1.0, mode="scrub")
```

## Benchmarking
See the Benchmarking docs for the full benchmark suite and reproducible runs:
[https://janclemenslab.org/acvr/benchmark/]().

### Scrub tuning
For tighter scrub accuracy across varied assets, prefer `keyframe_mode='nearest'` (the default)
and a smaller scrub bucket such as `scrub_bucket_ms=25`:
```python
from acvr import VideoReader
vr = VideoReader(video_file_name, scrub_bucket_ms=25)
preview = vr.read_frame(t_s=1.5, mode="scrub")  # nearest keyframe
```

### Color conversion and performance
All modes decode to RGB by default for consistent analysis of pixel values.
If you need slightly higher throughput and do not require RGB, pass `decode_rgb=False`
to `read_frame`, `read_frame_fast`, or `read_keyframe_at` to avoid the conversion.
This typically provides a marginal speedup and returns BGR for fast paths.

### Sequential reads
For dense, in-order processing, prefer sequential access:
```python
from acvr import VideoReader

with VideoReader(video_file_name) as vr:
    for frame in vr:  # sequential decoder
        pass
```
Sequential reads avoid per-frame seeking and are substantially faster than
random-access reads.

### Indexing semantics
- `accurate`: index is the decode-order frame count (exact on CFR; may differ from nominal timeline on VFR).
- `accurate_timeline`: index is treated as timeline index based on nominal FPS (guessed_rate), and an accurate seek is used at that timestamp.
- `fast`: approximate, timeline-oriented, low-latency access (good alignment on VFR).
- `scrub`: keyframes only; fastest previews.

## Documentation
The latest documentation lives at https://janclemenslab.org/acvr.

To build the docs locally:
```shell
pip install acvr[docs]
mkdocs serve
```

## Publishing
Build and upload the distribution to PyPI:
```shell
pip install flit
flit build
flit publish
```

## Test videos
The test video generator script `scripts/make_test_h264_videos.py` requires `ffmpeg`
to be available on your PATH. The script also needs OpenCV; install the dev extras
to pull in a headless build:
```shell
pip install acvr[dev]
```
