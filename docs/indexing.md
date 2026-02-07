# Indexing and Modes

This page explains how acvr interprets frame indices and how the different
read modes trade off latency and precision, especially for variable-frame-rate
(VFR) videos.

## Index policies

acvr supports two ways to interpret an index `i`:

- decode: `i` refers to the decode-order frame number in the stream. This is
  exact and deterministic across reads. On constant-frame-rate (CFR) videos,
  decode-order aligns with the nominal timeline. On VFR videos, decode-order
  may differ from the nominal timeline.

- timeline: `i` is mapped to a timestamp using the nominal frame rate
  (`guessed_rate` from PyAV when available). The reader then seeks accurately
  to the frame at/after that timestamp. This aligns better with the nominal
  timeline on VFR, but two assets with the same nominal rate might still differ
  due to encoder/gop structures.

You can choose the policy for array-style access via:

```python
from acvr import VideoReader

# Use timeline indexing globally for [] access (e.g., reader[100])
reader = VideoReader(path, index_policy="timeline")
frame = reader[100]  # mapped via nominal FPS to timestamp
```

Regardless of `index_policy`, you can always call mode-specific APIs.

## Read modes

- accurate: index is decode-order, or pass `t_s` to seek by timestamp.
- accurate_timeline: index is mapped to timestamp via nominal FPS; then performs
  the same accurate timestamp seeking as `accurate`.
- fast: approximate seek optimized for low latency; uses nominal timeline for
  good VFR alignment; exact on CFR for typical content.
- scrub: returns nearby keyframes; very fast and approximate; defaults to
  nearest keyframe and uses a small cache bucket for speed.
- sequential: decode in order using `read_next` or iteration; no seeking.

## Recommendations

- For full sequential processing (e.g., feature extraction), prefer `read_next`
  or `for frame in reader`, which uses the sequential decoder.
- CFR or strict decode-order tasks: use `accurate` (or default `index_policy='decode'`).
- VFR and UI preview: use `fast` or `scrub` (nearest). For exact reads tied to
  nominal timeline, use `accurate_timeline`.
- If your project thinks in “timeline frames”, set `index_policy='timeline'`
  on the reader to make `reader[i]` follow the nominal timeline.
