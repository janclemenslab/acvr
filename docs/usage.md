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

## Accurate timestamp reads
```python
from acvr import VideoReader

reader = VideoReader("/path/to/video.mp4", build_index=True)
frame = reader.read_frame_at(1.25)
reader.close()
```

## Fast scrubbing
```python
from acvr import VideoReader

reader = VideoReader("/path/to/video.mp4", build_index=True)
keyframe = reader.read_keyframe_at(2.0, mode="nearest")
reader.close()
```

## Selectable read modes
```python
from acvr import VideoReader

reader = VideoReader("/path/to/video.mp4", build_index=True)

# Accurate frame by index
frame = reader.read_frame(index=120, mode="accurate")

# Fast approximation
frame = reader.read_frame(index=120, mode="fast")

# Fast keyframe scrubbing
frame = reader.read_frame(t_s=2.0, mode="scrub", keyframe_mode="nearest")

reader.close()
```

## Benchmark guidance
Recent benchmarks on the bundled CFR/VFR test assets (M1-class laptop, PyAV 12.x):

| Mode | CFR fast (ms/frame) | CFR accuracy | VFR fast (ms/frame) | VFR accuracy |
| --- | --- | --- | --- | --- |
| Accurate | ~89 | exact | ~88 | matches PyAV reference |
| Scrub (keyframes) | ~2 | very approximate | ~2 | very approximate |
| Fast (PyAV) | ~12 | matches OpenCV | ~12 | matches OpenCV |

Use `fast` for interactive frame inspection, `accurate` for exact frame data,
and `scrub` for very fast keyframe previews.
