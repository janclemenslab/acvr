# Accurate video reader (acvr)
Video reader built around PyAV for frame-accurate seeking.

Supports:
- accurate, random-access retrieval of individual frames from videos encoded with modern codecs (H.264, H.265)
- works with variable-frame rate videos
- LRU
- Fast scrubbing mode

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

with VideoReader(video_file_name, build_index=True) as vr:
    accurate = vr.read_frame(index=100, mode="accurate")
    fast = vr.read_frame(index=100, mode="fast")
    scrub = vr.read_frame(t_s=1.0, mode="scrub", keyframe_mode="nearest")
```

## Benchmark snapshot
On the bundled CFR/VFR test assets (M1-class laptop, PyAV 12.x):

| Mode | CFR fast (ms/frame) | CFR accuracy | VFR fast (ms/frame) | VFR accuracy |
| --- | --- | --- | --- | --- |
| Accurate | ~89 | exact | ~88 | matches PyAV reference |
| Scrub (keyframes) | ~2 | very approximate | ~2 | very approximate |
| Fast (PyAV) | ~12 | matches OpenCV | ~12 | matches OpenCV |

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
