# Technical Notes: Frame-Accurate Video Reading in acvr

This document explains how acvr achieves frame-accurate reads with PyAV/FFmpeg,
why video access can be subtle, and how to choose the right mode for your
workflow. It complements the “Usage” and “Indexing & Modes” docs with the
underlying concepts.

## Core concepts

### Keyframes and GOPs
- **Keyframes (I-frames)** are independently decodable frames.
- **Inter frames (P/B)** depend on earlier (and sometimes later) frames.
- **GOP (Group of Pictures)** is the keyframe plus its dependent frames.

Because P/B frames depend on context, random access to an arbitrary frame must
first decode from a keyframe. Any “seek-to-frame” operation is therefore a
combination of:

1. Seek to a keyframe at or before the target.
2. Decode forward until the target frame/time is reached.

This is the root reason why naïve random access can be slow or inaccurate in
many video APIs.

### PTS, DTS, and time base
- **PTS (Presentation Timestamp)** is when a frame should appear on the
  timeline.
- **DTS (Decoding Timestamp)** is when a frame should be decoded.
- **Time base** is the unit in which PTS/DTS are expressed.

acvr uses PyAV/FFmpeg’s PTS values as the ground truth for time mapping. When
PTS is missing, acvr falls back to DTS, and if both are missing it synthesizes
monotonic timestamps during a full decode pass.

The conversion is:

```
time_s = (pts - start_pts) * time_base
```

Because PTS is the authoritative timeline signal, acvr prefers PTS-based
seeking rather than relying on frame index arithmetic alone.

### CFR vs VFR (constant vs variable frame rate)
- **CFR**: every frame advances the timeline by a constant duration.
- **VFR**: frame intervals vary; the nominal FPS is only an average.

In VFR assets, “frame 100” does not necessarily map to `100 / fps` seconds. The
timeline is encoded in PTS values, not in frame counts. acvr therefore exposes
two different indexing models: decode-order indexing and timeline-based
indexing.

### Decode order vs timeline order
Some codecs include B-frames, where decode order differs from presentation
order. acvr relies on PTS to locate frames on the presentation timeline and
separately maintains a decode-order index for deterministic frame-number reads.

## Indexing models in acvr

### Decode-order indexing
Decode-order indexing treats `index = 0` as “first decoded frame” and walks the
stream in the order frames are produced. This is deterministic and exact for
`VideoReader[i]` when `index_policy="decode"` or when you use the `accurate`
mode with an index.

If you literally care about “the 100th decoded frame” in a VFR asset, decode-
order indexing is the right choice and does not require timestamps.

acvr builds a decode-order lookup table (`frame_pts`) by decoding the stream
once. This lets it map index → PTS reliably, even for VFR or B-frame content.

### Timeline indexing
Timeline indexing treats `index` as a nominal frame on the timeline:

```
t_s = index / nominal_fps
```

The nominal FPS comes from `guessed_rate` when available (PyAV’s best effort
for VFR), falling back to `average_rate` or `base_rate`.

Timeline indexing aligns better with “frame N on the wall clock” for VFR
content, but it is still an approximation because the nominal FPS is not an
exact timeline definition. For true timestamps, use `read_frame_at(t_s)`.

## Why random access can be inaccurate

Random access is inherently approximate in many video APIs because:

- **Keyframe seeking**: decoders seek to a keyframe and decode forward, so
  returning the *exact* target depends on how the seek anchor is chosen.
- **PTS rounding**: timestamps are discrete in `time_base` units; mapping from
  seconds or indices can round up/down.
- **VFR timelines**: `index / fps` is not a reliable timestamp for VFR.
- **B-frames**: decode order differs from presentation order, so “frame index”
  is ambiguous without a defined model.

acvr’s accurate modes avoid these pitfalls by explicitly mapping index → PTS or
timestamp → PTS and decoding forward from a keyframe anchor.

## Access modes and what they do

### Sequential (iteration / read_next)
- **Purpose**: fastest possible full pass.
- **Mechanism**: uses a dedicated sequential decoder, no seeks.
- **Accuracy**: exact decode order; ideal for dense processing pipelines.

### Accurate
- **Input**: index in decode order, or `t_s` for timestamp reads.
- **Mechanism**: resolve index → PTS, then keyframe-seek + decode forward.
- **Accuracy**: frame-accurate for decode-order indexing; robust on CFR/VFR.

`VideoReader[i]` uses this behavior when `index_policy="decode"`.

### Accurate timeline
- **Input**: index interpreted on nominal timeline, or `t_s`.
- **Mechanism**: map index → `t_s` via nominal FPS, then accurate timestamp seek.
- **Accuracy**: aligns with timeline on VFR better than decode-order indexing.

### Fast
- **Input**: index or `t_s`.
- **Mechanism**: approximate seek (PyAV/OpenCV-like) using nominal FPS.
- **Accuracy**: good for interactive previews; not guaranteed frame-accurate.
- **Latency**: lowest for random access.

### Scrub
- **Input**: index or `t_s`.
- **Mechanism**: returns keyframes only, using a cached bucketed keyframe map.
- **Accuracy**: approximate; best for thumbnails or timeline scrubbing.
- **Dependency**: requires a built keyframe index (`build_index=True`).

## Timeline access and `index_policy`

`VideoReader` offers global indexing policy for `reader[i]`:

- `index_policy="decode"` (default): `i` means decode-order frame index.
- `index_policy="timeline"`: `i` means nominal timeline frame.

Use timeline policy only when your application treats “frame number” as a
position on a nominal timeline (e.g., UI frame counters for VFR content).

## Keyframe index and caches

### Keyframe index
`build_index=True` triggers a full packet scan to record keyframe timestamps.
This upfront cost can reduce per-seek latency because accurate modes can seek
to the nearest known keyframe instead of the raw target timestamp.

### Frame cache
acvr optionally caches decoded frames by PTS (`decoded_frame_cache_size`). This
helps repeated access to nearby frames or repeated seeks to the same timestamp.

### Scrub bucket cache
`scrub_bucket_ms` groups timestamps into coarse buckets for scrubbing. Smaller
buckets improve precision at the cost of more cache churn.

## Practical guidance

- **Need deterministic frame numbers?** Use `accurate` with decode-order
  indexing or `index_policy="decode"`.
- **Need timeline-consistent reads on VFR?** Use `accurate_timeline` or
  `read_frame_at(t_s)`.
- **Interactive UI preview?** Use `fast` or `scrub` (keyframes only).
- **Batch processing?** Use sequential iteration or `read_next`.

## Common pitfalls and how acvr avoids them

- **Off-by-one frames**: caused by rounding index → time or PTS rounding;
  accurate modes work in PTS units and return the first frame at/after target.
- **Broken timestamps**: some files have invalid PTS; acvr will raise if too
  many frames must be decoded to reach a timestamp (`max_decode_frames`).
- **VFR confusion**: decode-order indexing is not a timeline; use timeline
  modes when frame numbers are meant to track time.

## Choosing the right API

- `read_frame_at(t_s)`: best for exact timeline timestamp reads.
- `read_frame(index=..., mode="accurate")`: exact decode-order access.
- `read_frame(index=..., mode="accurate_timeline")`: timeline-aligned access.
- `read_frame(index=..., mode="fast")`: low latency, approximate.
- `read_frame(t_s=..., mode="scrub")`: keyframe scrubbing.
- `read_next()` / iteration: fastest sequential decode.

For detailed usage examples, see `docs/usage.md` and `docs/indexing.md`.
