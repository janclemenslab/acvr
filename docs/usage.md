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
