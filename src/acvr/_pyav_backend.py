"""PyAV-backed implementation details for acvr."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Union

import av
import numpy as np

Number = Union[int, float]


@dataclass(frozen=True)
class DecodedFrame:
    """Container for decoded frame metadata and pixel data."""

    image: np.ndarray
    pts: Optional[int]
    time_s: float
    key_frame: bool


@dataclass(frozen=True)
class KeyframeEntry:
    """Metadata for a keyframe in the stream."""

    pts: int
    time_s: float


class _LRU:
    """Simple LRU with dict + order list."""

    def __init__(self, capacity: int) -> None:
        """Create an LRU with a fixed capacity."""

        self.capacity = max(0, int(capacity))
        self._data: Dict[int, object] = {}
        self._order: List[int] = []

    def get(self, key: int) -> Optional[object]:
        """Return an item and mark it as most recently used."""

        value = self._data.get(key)
        if value is None:
            return None
        try:
            self._order.remove(key)
        except ValueError:
            pass
        self._order.append(key)
        return value

    def put(self, key: int, value: object) -> None:
        """Insert or update an item in the cache."""

        if self.capacity <= 0:
            return
        if key in self._data:
            self._data[key] = value
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return
        if len(self._data) >= self.capacity:
            old = self._order.pop(0)
            self._data.pop(old, None)
        self._data[key] = value
        self._order.append(key)

    def clear(self) -> None:
        """Clear all entries from the cache."""

        self._data.clear()
        self._order.clear()


class PyAVVideoBackend:
    """Frame-accurate seeking with keyframe index and scrub acceleration."""

    def __init__(
        self,
        path: str,
        video_stream_index: int = 0,
        *,
        build_index: bool = False,
        decoded_frame_cache_size: int = 0,
        scrub_bucket_ms: int = 100,
        scrub_bucket_lru_size: int = 4096,
    ) -> None:
        """Initialize the PyAV-backed decoder."""

        self._path = path
        self._container = av.open(path)
        self._stream = self._container.streams.video[video_stream_index]
        self._codec_ctx = self._stream.codec_context

        self._time_base: Fraction = self._stream.time_base
        self._start_pts: int = self._stream.start_time if self._stream.start_time is not None else 0

        self._keyframes: List[KeyframeEntry] = []
        self._index_built: bool = False

        self._frame_pts: Optional[List[int]] = None
        self._frame_count: int = int(self._stream.frames or 0)
        self._current_frame_pos: float = 0.0

        self._frame_cache = _LRU(decoded_frame_cache_size)

        self._scrub_bucket_ms = max(1, int(scrub_bucket_ms))
        self._bucket_to_kfidx = _LRU(scrub_bucket_lru_size)

        if build_index:
            self.build_keyframe_index()

        self._frame_height = int(self._stream.height or 0)
        self._frame_width = int(self._stream.width or 0)
        self._frame_shape = (self._frame_height, self._frame_width, 3)
        self._frame_rate = self._compute_frame_rate()
        self._fourcc = self._compute_fourcc()
        self._frame_format = 0

    def close(self) -> None:
        """Close the underlying PyAV container."""

        self._container.close()

    def __enter__(self) -> "PyAVVideoBackend":
        """Return self for context manager usage."""

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the backend on exit from a context manager."""

        self.close()

    def _secs_to_pts(self, t_s: float) -> int:
        """Convert seconds to presentation timestamp units."""

        ticks = int(round(t_s / float(self._time_base)))
        return self._start_pts + ticks

    def _pts_to_secs(self, pts: int) -> float:
        """Convert presentation timestamp units to seconds."""

        return float((pts - self._start_pts) * self._time_base)

    def _frame_time_s(self, pts: Optional[int]) -> float:
        """Return the timestamp for a frame PTS."""

        return float("nan") if pts is None else self._pts_to_secs(pts)

    def _flush_decoder(self) -> None:
        """Flush decoder buffers if supported."""

        try:
            self._codec_ctx.flush_buffers()
        except Exception:
            pass

    def _compute_frame_rate(self) -> float:
        """Compute the stream frame rate in frames per second."""

        rate = self._stream.average_rate or self._stream.base_rate
        return float(rate) if rate is not None else 0.0

    def _compute_fourcc(self) -> int:
        """Compute a fourcc code from the stream codec tag."""

        tag = self._stream.codec_context.codec_tag
        if isinstance(tag, str) and len(tag) >= 4:
            tag = tag[:4]
            return (
                ord(tag[0])
                | (ord(tag[1]) << 8)
                | (ord(tag[2]) << 16)
                | (ord(tag[3]) << 24)
            )
        return 0

    def _ensure_frame_pts(self) -> None:
        """Decode the stream once to collect frame PTS values."""

        if self._frame_pts is not None:
            return
        idx_container = av.open(self._path)
        idx_stream = idx_container.streams.video[self._stream.index]
        pts_list: List[int] = []
        for frame in idx_container.decode(idx_stream):
            pts = frame.pts if frame.pts is not None else frame.dts
            if pts is None:
                pts = self._start_pts + len(pts_list)
            pts_list.append(int(pts))
        idx_container.close()
        self._frame_pts = pts_list
        self._frame_count = len(pts_list)

    def _read_frame_by_pts(self, target_pts: int) -> DecodedFrame:
        """Decode the first frame at or after a target PTS."""

        cached = self._frame_cache.get(target_pts)
        if cached is not None:
            return cached  # type: ignore[return-value]

        if self._index_built:
            idx = self._keyframe_index_at_or_before_pts(target_pts)
            seek_pts = self._keyframes[idx].pts
        else:
            seek_pts = target_pts

        container = av.open(self._path)
        stream = container.streams.video[self._stream.index]
        try:
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            try:
                stream.codec_context.flush_buffers()
            except Exception:
                pass

            last: Optional[DecodedFrame] = None
            for packet in container.demux(stream):
                for frame in packet.decode():
                    pts = frame.pts
                    cur = DecodedFrame(
                        image=frame.to_rgb().to_ndarray(),
                        pts=pts,
                        time_s=self._frame_time_s(pts),
                        key_frame=bool(getattr(frame, "key_frame", False)),
                    )
                    if pts is not None:
                        self._frame_cache.put(int(pts), cur)
                    if pts is None:
                        last = cur
                        continue
                    if pts >= target_pts:
                        return cur
                    last = cur
        finally:
            container.close()

        if last is not None:
            return last
        raise RuntimeError("Could not decode any frames after seeking.")

    def frame_at_index(self, index: int) -> np.ndarray:
        """Return the decoded frame at a zero-based index."""

        self._ensure_frame_pts()
        assert self._frame_pts is not None
        if index < 0:
            index += self._frame_count
        if index < 0 or index >= self._frame_count:
            raise IndexError("frame index out of range")
        target_pts = self._frame_pts[index]
        decoded = self._read_frame_by_pts(target_pts)
        self._current_frame_pos = float(index)
        return decoded.image

    @property
    def frame_height(self) -> int:
        """Return the video frame height."""

        return self._frame_height

    @property
    def frame_width(self) -> int:
        """Return the video frame width."""

        return self._frame_width

    @property
    def frame_rate(self) -> float:
        """Return the reported frame rate in frames per second."""

        return self._frame_rate

    @property
    def fourcc(self) -> int:
        """Return the fourcc codec identifier."""

        return self._fourcc

    @property
    def frame_format(self) -> int:
        """Return the frame format identifier."""

        return self._frame_format

    @property
    def number_of_frames(self) -> int:
        """Return the total number of frames, decoding if needed."""

        if self._frame_count <= 0:
            self._ensure_frame_pts()
        return self._frame_count

    @property
    def frame_shape(self) -> tuple:
        """Return the expected frame shape (H, W, C)."""

        return self._frame_shape

    @property
    def current_frame_pos(self) -> float:
        """Return the last frame index accessed."""

        return self._current_frame_pos

    def _seek_to_pts(self, pts: int, *, backward: bool) -> None:
        """Seek to a timestamp in the stream."""

        self._container.seek(pts, stream=self._stream, backward=backward, any_frame=False)
        self._flush_decoder()

    def build_keyframe_index(self, *, max_packets: Optional[int] = None) -> List[KeyframeEntry]:
        """Scan packets and store keyframe pts/time."""

        path = self._container.name
        idx_container = av.open(path)
        idx_stream = idx_container.streams.video[self._stream.index]

        key_pts: List[int] = []
        n = 0
        for packet in idx_container.demux(idx_stream):
            if packet.dts is None and packet.pts is None:
                continue
            if packet.is_keyframe:
                pts = packet.pts if packet.pts is not None else packet.dts
                if pts is not None:
                    key_pts.append(int(pts))
            n += 1
            if max_packets is not None and n >= max_packets:
                break

        idx_container.close()

        key_pts = sorted(set(key_pts))
        if not key_pts:
            key_pts = [self._start_pts]

        self._keyframes = [KeyframeEntry(pts=p, time_s=self._pts_to_secs(p)) for p in key_pts]
        self._index_built = True

        self._bucket_to_kfidx.clear()
        return self._keyframes

    def _keyframe_index_at_or_before_pts(self, target_pts: int) -> int:
        """Return keyframe index at or before the target PTS."""

        kf = self._keyframes
        if not self._index_built or not kf:
            return 0
        if target_pts <= kf[0].pts:
            return 0
        if target_pts >= kf[-1].pts:
            return len(kf) - 1

        lo, hi = 0, len(kf) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            m = kf[mid].pts
            if m == target_pts:
                return mid
            if m < target_pts:
                lo = mid + 1
            else:
                hi = mid - 1
        return hi

    def _keyframe_index_nearest_pts(self, target_pts: int) -> int:
        """Return nearest keyframe index to the target PTS."""

        kf = self._keyframes
        if not self._index_built or not kf:
            return 0
        i0 = self._keyframe_index_at_or_before_pts(target_pts)
        i1 = min(i0 + 1, len(kf) - 1)
        if i0 == i1:
            return i0
        d0 = abs(kf[i0].pts - target_pts)
        d1 = abs(kf[i1].pts - target_pts)
        return i0 if d0 <= d1 else i1

    def _bucket_key(self, t_s: float) -> int:
        """Return a bucket key for the scrub acceleration cache."""

        return int(round(t_s * 1000.0 / self._scrub_bucket_ms))

    def _keyframe_index_for_time_fast(self, t_s: float, mode: str) -> int:
        """Return a keyframe index using cached time buckets."""

        if not self._index_built:
            raise RuntimeError("Keyframe index not built. Call build_keyframe_index() first.")

        b = self._bucket_key(t_s)

        mode_tag = {"previous": 0, "nearest": 1, "next": 2}.get(mode)
        if mode_tag is None:
            raise ValueError("mode must be one of: 'previous', 'nearest', 'next'")
        cache_key = (b << 2) | mode_tag

        cached = self._bucket_to_kfidx.get(cache_key)
        if cached is not None:
            return int(cached)

        target_pts = self._secs_to_pts(t_s)

        if mode == "previous":
            idx = self._keyframe_index_at_or_before_pts(target_pts)
        elif mode == "nearest":
            idx = self._keyframe_index_nearest_pts(target_pts)
        else:
            i_prev = self._keyframe_index_at_or_before_pts(target_pts)
            if self._keyframes[i_prev].pts >= target_pts:
                idx = i_prev
            else:
                idx = min(i_prev + 1, len(self._keyframes) - 1)

        self._bucket_to_kfidx.put(cache_key, idx)
        return idx

    def read_keyframe_at(
        self,
        t_s: Number,
        *,
        mode: str = "previous",
        decode_rgb: bool = True,
    ) -> DecodedFrame:
        """Return a nearby keyframe without GOP forward decoding."""

        t_s = float(t_s)
        idx = self._keyframe_index_for_time_fast(t_s, mode)
        key_pts = self._keyframes[idx].pts

        cached = self._frame_cache.get(key_pts)
        if cached is not None:
            return cached  # type: ignore[return-value]

        self._seek_to_pts(key_pts, backward=False)

        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                pts = frame.pts
                img = frame.to_rgb().to_ndarray() if decode_rgb else frame.to_ndarray()
                cur = DecodedFrame(
                    image=img,
                    pts=pts,
                    time_s=self._frame_time_s(pts),
                    key_frame=bool(getattr(frame, "key_frame", False)),
                )
                if pts is not None:
                    self._frame_cache.put(int(pts), cur)
                return cur

        raise RuntimeError("Failed to decode a frame after keyframe seek.")

    def read_frame_at(
        self,
        t_s: Number,
        *,
        return_first_after: bool = True,
        max_decode_frames: int = 10_000,
        use_index: bool = True,
    ) -> DecodedFrame:
        """Decode a frame near a timestamp with accurate seeking."""

        t_s = float(t_s)
        target_pts = self._secs_to_pts(t_s)

        cached = self._frame_cache.get(target_pts)
        if cached is not None:
            return cached  # type: ignore[return-value]

        if use_index and self._index_built:
            idx = self._keyframe_index_at_or_before_pts(target_pts)
            anchor_pts = self._keyframes[idx].pts
            self._seek_to_pts(anchor_pts, backward=False)
        else:
            self._seek_to_pts(target_pts, backward=True)

        last: Optional[DecodedFrame] = None
        decoded = 0

        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                decoded += 1
                if decoded > max_decode_frames:
                    raise RuntimeError(
                        "Exceeded max_decode_frames while seeking; timestamps may be broken."
                    )

                pts = frame.pts
                cur = DecodedFrame(
                    image=frame.to_rgb().to_ndarray(),
                    pts=pts,
                    time_s=self._frame_time_s(pts),
                    key_frame=bool(getattr(frame, "key_frame", False)),
                )
                if pts is not None:
                    self._frame_cache.put(int(pts), cur)

                if pts is None:
                    last = cur
                    continue

                if return_first_after:
                    if pts >= target_pts:
                        return cur
                    last = cur
                else:
                    if pts <= target_pts:
                        last = cur
                    elif last is not None:
                        return last

        if last is not None:
            return last
        raise RuntimeError("Could not decode any frames after seeking.")
