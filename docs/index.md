# acvr

acvr provides frame-accurate video access with an array-style interface. It is built on
PyAV and focuses on reliable random access for modern codecs (H.264/H.265), including
variable-frame-rate assets.

## Highlights
- Array-style access (`reader[100]`) for frames
- Iteration over frames (`for frame in reader`)
- Context manager support (`with VideoReader(path) as reader`)
- Accurate seeking using timestamps and keyframe indexes

## Installation
```shell
pip install acvr
```

For development extras (OpenCV + pytest):
```shell
pip install acvr[dev]
```
