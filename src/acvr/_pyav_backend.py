"""PyAV-backed implementation details for acvr."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterator, List, Optional, Union

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
        scrub_bucket_ms: int = 25,
        scrub_bucket_lru_size: int = 4096,
        threading: bool = True,
        thread_count: int = 0,
    ) -> None:
        """Initialize the PyAV-backed decoder."""

        self._path = path
        self._container = av.open(path)
        self._stream = self._container.streams.video[video_stream_index]
        self._codec_ctx = self._stream.codec_context
        self._fast_container: Optional[av.container.InputContainer] = None
        self._fast_stream: Optional[av.video.stream.VideoStream] = None
        self._fast_first_frame_number: Optional[int] = None
        self._fast_decoder = None
        self._fast_last_pts: Optional[int] = None
        self._seq_container: Optional[av.container.InputContainer] = None
        self._seq_stream: Optional[av.video.stream.VideoStream] = None
        self._seq_decoder = None
        self._seq_frame_index: int = 0
        self._last_index: Optional[int] = None
        self._last_fast_index: Optional[int] = None

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
        self._threading = bool(threading)
        self._thread_count = int(thread_count)

        if build_index:
            self.build_keyframe_index()

        self._frame_height = int(self._stream.height or 0)
        self._frame_width = int(self._stream.width or 0)
        self._frame_shape = (self._frame_height, self._frame_width, 3)
        self._frame_rate = self._compute_frame_rate()
        self._nominal_frame_rate = self._compute_nominal_frame_rate()
        self._fourcc = self._compute_fourcc()
        self._frame_format = 0

    def close(self) -> None:
        """Close the underlying PyAV container."""

        self._container.close()
        if self._fast_container is not None:
            self._fast_container.close()
            self._fast_container = None
            self._fast_stream = None
            self._fast_decoder = None
            self._fast_last_pts = None
        self._fast_first_frame_number = None
        if self._seq_container is not None:
            self._seq_container.close()
            self._seq_container = None
            self._seq_stream = None
            self._seq_decoder = None
        self._seq_frame_index = 0
        self._last_index = None
        self._last_fast_index = None

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

    def _pts_to_frame_number(self, pts: Optional[int], fps: float) -> Optional[int]:
        """Convert a PTS value to a rounded frame number."""

        if pts is None:
            return None
        return int(round(self._pts_to_secs(int(pts)) * fps))

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

    def _compute_nominal_frame_rate(self) -> float:
        """Compute a nominal frame rate preferring guessed_rate (useful for VFR)."""

        rate = getattr(self._stream, "guessed_rate", None) or self._stream.average_rate or self._stream.base_rate
        return float(rate) if rate is not None else 0.0

    @property
    def nominal_frame_rate(self) -> float:
        """Return the nominal frame rate (guessed_rate when available)."""

        return self._nominal_frame_rate

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
        self._configure_codec_context(idx_stream)
        self._configure_codec_context(idx_stream)
        pts_list: List[int] = []
        for frame in idx_container.decode(idx_stream):
            pts = frame.pts if frame.pts is not None else frame.dts
            if pts is None:
                pts = self._start_pts + len(pts_list)
            pts_list.append(int(pts))
        idx_container.close()
        self._frame_pts = pts_list
        self._frame_count = len(pts_list)

    def _read_frame_by_pts(self, target_pts: int, *, decode_rgb: bool = True) -> DecodedFrame:
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
        self._configure_codec_context(stream)
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
                    image = frame.to_rgb().to_ndarray() if decode_rgb else frame.to_ndarray(format="bgr24")
                    cur = DecodedFrame(
                        image=image,
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

    def _read_frame_at_index(
        self,
        index: int,
        *,
        decode_rgb: bool = True,
        use_sequential: bool = True,
    ) -> DecodedFrame:
        """Return the decoded frame at a zero-based index."""

        if use_sequential and index >= 0:
            if self._seq_decoder is not None and index == self._seq_frame_index:
                return self.read_next_frame(decode_rgb=decode_rgb)
            if self._seq_decoder is None and index == 0:
                self.reset_sequence()
                return self.read_next_frame(decode_rgb=decode_rgb)

        self._ensure_frame_pts()
        assert self._frame_pts is not None
        if index < 0:
            index += self._frame_count
        if index < 0 or index >= self._frame_count:
            raise IndexError("frame index out of range")
        target_pts = self._frame_pts[index]
        if use_sequential and self._seq_decoder is not None and index == self._seq_frame_index:
            try:
                decoded = self.read_next_frame(decode_rgb=decode_rgb)
            except StopIteration:
                decoded = self._seek_seq_to_pts(target_pts, target_index=index, decode_rgb=decode_rgb)
        elif use_sequential and self._last_index is not None and index == self._last_index + 1:
            decoded = self._seek_seq_to_pts(target_pts, target_index=index, decode_rgb=decode_rgb)
        else:
            decoded = self._read_frame_by_pts(target_pts, decode_rgb=decode_rgb)
            self._current_frame_pos = float(index)
        self._last_index = index
        return decoded

    def frame_at_index(self, index: int) -> np.ndarray:
        """Return the decoded frame at a zero-based index."""

        return self._read_frame_at_index(index, decode_rgb=True, use_sequential=True).image

    #
    # Public helpers for PTS/time mapping
    #
    def pts_at_index(self, index: int) -> Optional[int]:
        """Return the presentation timestamp (PTS) for a frame index."""

        self._ensure_frame_pts()
        assert self._frame_pts is not None
        if index < 0:
            index += self._frame_count
        if index < 0 or index >= self._frame_count:
            raise IndexError("frame index out of range")
        return int(self._frame_pts[index])

    def time_at_index(self, index: int) -> float:
        """Return the timestamp in seconds for a frame index."""

        pts = self.pts_at_index(index)
        return float("nan") if pts is None else self._pts_to_secs(int(pts))

    def index_from_pts(self, pts: int) -> int:
        """Map a PTS value to the nearest frame index."""

        self._ensure_frame_pts()
        assert self._frame_pts is not None
        fps = self._frame_pts
        if not fps:
            return 0
        lo, hi = 0, len(fps) - 1
        if pts <= fps[0]:
            return 0
        if pts >= fps[-1]:
            return hi
        while lo <= hi:
            mid = (lo + hi) // 2
            m = fps[mid]
            if m == pts:
                return mid
            if m < pts:
                lo = mid + 1
            else:
                hi = mid - 1
        # choose nearest between hi and lo
        if lo >= len(fps):
            return hi
        if hi < 0:
            return lo
        return lo if abs(fps[lo] - pts) < abs(fps[hi] - pts) else hi

    def index_from_time(self, t_s: float) -> int:
        """Map a timestamp in seconds to the nearest frame index."""

        pts = self._secs_to_pts(float(t_s))
        return self.index_from_pts(int(pts))

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

    def _ensure_fast_container(self) -> None:
        """Initialize the fast-seek container if needed."""

        if self._fast_container is not None:
            return
        self._fast_container = av.open(self._path)
        self._fast_stream = self._fast_container.streams.video[self._stream.index]
        self._configure_codec_context(self._fast_stream)

    def _configure_codec_context(self, stream: av.video.stream.VideoStream) -> None:
        """Apply conservative threading settings to a codec context."""

        try:
            codec_ctx = stream.codec_context
            if self._threading:
                codec_ctx.thread_type = "AUTO"
                codec_ctx.thread_count = self._thread_count
            else:
                codec_ctx.thread_type = "NONE"
                codec_ctx.thread_count = 1
        except Exception:
            pass

    def _fast_rewind(self) -> None:
        """Rewind the fast-seek container and reset decoder state."""

        self._ensure_fast_container()
        assert self._fast_container is not None
        assert self._fast_stream is not None
        self._fast_container.seek(0)
        self._fast_decoder = self._fast_container.decode(self._fast_stream)
        self._fast_last_pts = None

    def _fast_frame_to_pts(self, frame_index: int, fps: float) -> int:
        """Convert a frame index to expected PTS for fast reads."""

        if frame_index <= 0:
            return self._start_pts
        return self._secs_to_pts(frame_index / fps)

    def _read_frame_fast_like(self, target_frame: int, *, decode_rgb: bool) -> DecodedFrame:
        """Approximate FastVideoReader behavior for fast sequential reads."""

        fps = self._nominal_frame_rate or self._frame_rate or 1.0
        pts_per_frame = 1.0 / (fps * float(self._time_base)) if fps else 1.0
        wiggle = pts_per_frame / 10.0

        if target_frame <= 0:
            self._fast_rewind()
            assert self._fast_decoder is not None
            frame = next(self._fast_decoder)
            pts = frame.pts if frame.pts is not None else frame.dts
            if pts is None:
                pts = self._fast_frame_to_pts(0, fps)
            img = frame.to_rgb().to_ndarray() if decode_rgb else frame.to_ndarray(format="bgr24")
            cur = DecodedFrame(
                image=img,
                pts=pts,
                time_s=self._frame_time_s(pts),
                key_frame=bool(getattr(frame, "key_frame", False)),
            )
            if pts is not None:
                self._frame_cache.put(int(pts), cur)
            return cur

        expected_prev_pts = self._fast_frame_to_pts(target_frame - 1, fps)
        if self._fast_decoder is not None and self._fast_last_pts == expected_prev_pts:
            try:
                frame = next(self._fast_decoder)
            except StopIteration:
                frame = None
            if frame is not None:
                pts = frame.pts if frame.pts is not None else frame.dts
                if pts is None:
                    pts = self._fast_frame_to_pts(target_frame, fps)
                img = frame.to_rgb().to_ndarray() if decode_rgb else frame.to_ndarray(format="bgr24")
                cur = DecodedFrame(
                    image=img,
                    pts=pts,
                    time_s=self._frame_time_s(pts),
                    key_frame=bool(getattr(frame, "key_frame", False)),
                )
                if pts is not None:
                    self._frame_cache.put(int(pts), cur)
                self._fast_last_pts = int(pts) if pts is not None else None
                return cur

        self._ensure_fast_container()
        assert self._fast_container is not None
        assert self._fast_stream is not None

        target_pts = self._fast_frame_to_pts(target_frame, fps)
        self._fast_container.seek(
            target_pts,
            stream=self._fast_stream,
            backward=True,
            any_frame=False,
        )
        try:
            self._fast_stream.codec_context.flush_buffers()
        except Exception:
            pass
        self._fast_decoder = self._fast_container.decode(self._fast_stream)
        self._fast_last_pts = None

        try:
            frame = next(self._fast_decoder)
        except StopIteration:
            frame = None
        if frame is None:
            return self._read_frame_fast_simple(target_pts, decode_rgb=decode_rgb)

        cur_pts = frame.pts if frame.pts is not None else frame.dts
        if cur_pts is None:
            cur_pts = target_pts

        if cur_pts > target_pts:
            back = max(1, int(round(100)))
            back_pts = self._fast_frame_to_pts(max(0, target_frame - back), fps)
            self._fast_container.seek(
                back_pts,
                stream=self._fast_stream,
                backward=True,
                any_frame=False,
            )
            try:
                self._fast_stream.codec_context.flush_buffers()
            except Exception:
                pass
            self._fast_decoder = self._fast_container.decode(self._fast_stream)
            try:
                frame = next(self._fast_decoder)
            except StopIteration:
                frame = None
            if frame is None:
                return self._read_frame_fast_simple(target_pts, decode_rgb=decode_rgb)
            cur_pts = frame.pts if frame.pts is not None else frame.dts
            if cur_pts is None:
                cur_pts = target_pts

        while float(cur_pts) < (float(target_pts) - wiggle):
            try:
                frame = next(self._fast_decoder)
            except StopIteration:
                frame = None
                break
            if frame is None:
                break
            cur_pts = frame.pts if frame.pts is not None else frame.dts
            if cur_pts is None:
                cur_pts = target_pts
                break

        if frame is None:
            return self._read_frame_fast_simple(target_pts, decode_rgb=decode_rgb)

        img = frame.to_rgb().to_ndarray() if decode_rgb else frame.to_ndarray(format="bgr24")
        cur = DecodedFrame(
            image=img,
            pts=cur_pts,
            time_s=self._frame_time_s(cur_pts),
            key_frame=bool(getattr(frame, "key_frame", False)),
        )
        if cur_pts is not None:
            self._frame_cache.put(int(cur_pts), cur)
            self._fast_last_pts = int(cur_pts)
        else:
            self._fast_last_pts = None
        return cur

    def _ensure_seq_container(self) -> None:
        """Initialize a sequential decode container if needed."""

        if self._seq_container is not None:
            return
        self._seq_container = av.open(self._path)
        self._seq_stream = self._seq_container.streams.video[self._stream.index]
        self._configure_codec_context(self._seq_stream)
        self._seq_decoder = self._seq_container.decode(self._seq_stream)
        self._seq_frame_index = 0

    def reset_sequence(self) -> None:
        """Reset sequential decoding to the first frame."""

        self._ensure_seq_container()
        assert self._seq_container is not None
        assert self._seq_stream is not None
        try:
            self._seq_container.seek(0)
        except Exception:
            pass
        self._seq_decoder = self._seq_container.decode(self._seq_stream)
        self._seq_frame_index = 0

    def _seek_seq_to_pts(
        self,
        target_pts: int,
        *,
        target_index: int,
        decode_rgb: bool,
        any_frame: bool = False,
    ) -> DecodedFrame:
        """Seek the sequential decoder to a PTS and return the first match."""

        self._ensure_seq_container()
        assert self._seq_container is not None
        assert self._seq_stream is not None

        seek_pts = target_pts
        if self._keyframes and not any_frame:
            idx = self._keyframe_index_at_or_before_pts(target_pts)
            seek_pts = self._keyframes[idx].pts

        self._seq_container.seek(
            seek_pts,
            stream=self._seq_stream,
            backward=True,
            any_frame=any_frame,
        )
        try:
            self._seq_stream.codec_context.flush_buffers()
        except Exception:
            pass

        decoder = self._seq_container.decode(self._seq_stream)
        last: Optional[DecodedFrame] = None
        for frame in decoder:
            pts = frame.pts if frame.pts is not None else frame.dts
            img = frame.to_rgb().to_ndarray() if decode_rgb else frame.to_ndarray(format="bgr24")
            cur = DecodedFrame(
                image=img,
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
                self._seq_decoder = decoder
                self._seq_frame_index = target_index + 1
                self._current_frame_pos = float(target_index)
                return cur
            last = cur

        if last is not None:
            self._seq_decoder = decoder
            self._seq_frame_index = target_index + 1
            self._current_frame_pos = float(target_index)
            return last
        raise RuntimeError("Could not decode any frames after seeking.")

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
        mode: str = "nearest",
        decode_rgb: bool = True,
    ) -> DecodedFrame:
        """Return a nearby keyframe without GOP forward decoding."""

        t_s = float(t_s)
        idx = self._keyframe_index_for_time_fast(t_s, mode)
        key_pts = self._keyframes[idx].pts

        cached = self._frame_cache.get(key_pts)
        if cached is not None:
            return cached  # type: ignore[return-value]

        # Use a fresh container for reliable keyframe seek, avoiding stateful issues
        container = av.open(self._path)
        try:
            stream = container.streams.video[self._stream.index]
            self._configure_codec_context(stream)
            # Use backward seek to land on or before the requested keyframe PTS reliably
            container.seek(key_pts, stream=stream, backward=True, any_frame=False)
            try:
                stream.codec_context.flush_buffers()
            except Exception:
                pass

            for packet in container.demux(stream):
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
        finally:
            container.close()

        raise RuntimeError("Failed to decode a frame after keyframe seek.")

    def read_frame_at(
        self,
        t_s: Number,
        *,
        return_first_after: bool = True,
        max_decode_frames: int = 10_000,
        use_index: bool = True,
    ) -> DecodedFrame:
        """Decode a frame near a timestamp with accurate seeking.

        Simplified and robust: always uses a fresh container and backward keyframe seek.
        """

        t_s = float(t_s)
        target_pts = self._secs_to_pts(t_s)

        cached = self._frame_cache.get(target_pts)
        if cached is not None:
            return cached  # type: ignore[return-value]

        container = av.open(self._path)
        try:
            stream = container.streams.video[self._stream.index]
            self._configure_codec_context(stream)
            if use_index and self._index_built:
                idx = self._keyframe_index_at_or_before_pts(target_pts)
                anchor_pts = self._keyframes[idx].pts
                container.seek(anchor_pts, stream=stream, backward=True, any_frame=False)
            else:
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)
            try:
                stream.codec_context.flush_buffers()
            except Exception:
                pass

            last: Optional[DecodedFrame] = None
            decoded = 0
            for packet in container.demux(stream):
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
        finally:
            container.close()

    def _read_frame_fast_simple(self, target_pts: int, *, decode_rgb: bool) -> DecodedFrame:
        """Fallback fast seek: seek and decode first frame after PTS."""

        self._ensure_fast_container()
        assert self._fast_container is not None
        assert self._fast_stream is not None

        def grab_frame(container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> Optional[DecodedFrame]:
            for frame in container.decode(stream):
                pts = frame.pts if frame.pts is not None else frame.dts
                if pts is None:
                    target_reached = True
                else:
                    target_reached = pts >= target_pts
                if not target_reached:
                    continue
                if decode_rgb:
                    img = frame.to_rgb().to_ndarray()
                else:
                    img = frame.to_ndarray(format="bgr24")
                cur = DecodedFrame(
                    image=img,
                    pts=pts,
                    time_s=self._frame_time_s(pts),
                    key_frame=bool(getattr(frame, "key_frame", False)),
                )
                if pts is not None:
                    self._frame_cache.put(int(pts), cur)
                return cur
            return None

        self._fast_container.seek(
            target_pts,
            stream=self._fast_stream,
            backward=True,
            any_frame=True,
        )
        try:
            self._fast_stream.codec_context.flush_buffers()
        except Exception:
            pass
        grabbed = grab_frame(self._fast_container, self._fast_stream)
        if grabbed is not None:
            return grabbed

        self._fast_container.seek(
            target_pts,
            stream=self._fast_stream,
            backward=True,
            any_frame=False,
        )
        try:
            self._fast_stream.codec_context.flush_buffers()
        except Exception:
            pass
        grabbed = grab_frame(self._fast_container, self._fast_stream)
        if grabbed is not None:
            return grabbed

        raise RuntimeError("Failed to decode a frame after fast seek.")

    def _read_frame_fast_opencv_pyav(self, target_frame: int, *, decode_rgb: bool) -> DecodedFrame:
        """Approximate OpenCV seek behavior using PyAV."""

        fps = self._nominal_frame_rate or self._frame_rate or 1.0
        self._ensure_fast_container()
        assert self._fast_container is not None
        assert self._fast_stream is not None

        def seek_to_frame(frame_index: int) -> None:
            target_pts = self._secs_to_pts(frame_index / fps)
            self._fast_container.seek(
                target_pts,
                stream=self._fast_stream,
                backward=True,
                any_frame=False,
            )
            try:
                self._fast_stream.codec_context.flush_buffers()
            except Exception:
                pass

        def frame_number_from_pts(pts: Optional[int]) -> Optional[int]:
            num = self._pts_to_frame_number(pts, fps)
            if num is None:
                return None
            if self._fast_first_frame_number is None:
                return num
            return num - self._fast_first_frame_number

        first_frame = None
        if self._fast_first_frame_number is None:
            seek_to_frame(0)
            for frame in self._fast_container.decode(self._fast_stream):
                pts = frame.pts if frame.pts is not None else frame.dts
                self._fast_first_frame_number = self._pts_to_frame_number(pts, fps) or 0
                first_frame = frame
                break

        if target_frame <= 0:
            if first_frame is None:
                seek_to_frame(0)
                for frame in self._fast_container.decode(self._fast_stream):
                    first_frame = frame
                    break
            if first_frame is None:
                raise RuntimeError("Failed to decode a frame after fast seek.")
            pts = first_frame.pts if first_frame.pts is not None else first_frame.dts
            img = first_frame.to_rgb().to_ndarray() if decode_rgb else first_frame.to_ndarray(format="bgr24")
            cur = DecodedFrame(
                image=img,
                pts=pts,
                time_s=self._frame_time_s(pts),
                key_frame=bool(getattr(first_frame, "key_frame", False)),
            )
            if pts is not None:
                self._frame_cache.put(int(pts), cur)
            return cur

        delta = 16
        attempts = 0
        while True:
            start_frame = max(target_frame - delta, 0)
            seek_to_frame(start_frame)
            decoder = self._fast_container.decode(self._fast_stream)
            try:
                frame = next(decoder)
            except StopIteration:
                break

            pts = frame.pts if frame.pts is not None else frame.dts
            frame_number = frame_number_from_pts(pts)
            if frame_number is None:
                frame_number = start_frame

            if frame_number < 0 or frame_number > target_frame:
                if start_frame == 0 or delta >= 1 << 30 or attempts > 20:
                    break
                delta = delta * 2 if delta < 16 else int(delta * 1.5)
                attempts += 1
                continue

            while frame_number < target_frame:
                try:
                    frame = next(decoder)
                except StopIteration:
                    frame = None
                    break
                pts = frame.pts if frame.pts is not None else frame.dts
                frame_number = frame_number_from_pts(pts)
                if frame_number is None:
                    frame_number = target_frame

            if frame is None:
                break

            pts = frame.pts if frame.pts is not None else frame.dts
            img = frame.to_rgb().to_ndarray() if decode_rgb else frame.to_ndarray(format="bgr24")
            cur = DecodedFrame(
                image=img,
                pts=pts,
                time_s=self._frame_time_s(pts),
                key_frame=bool(getattr(frame, "key_frame", False)),
            )
            if pts is not None:
                self._frame_cache.put(int(pts), cur)
            return cur

        target_pts = self._secs_to_pts(target_frame / fps)
        return self._read_frame_fast_simple(target_pts, decode_rgb=decode_rgb)


    def read_frame_fast(
        self,
        *,
        index: Optional[int] = None,
        t_s: Optional[Number] = None,
        decode_rgb: bool = True,
        use_sequential: bool = True,
    ) -> DecodedFrame:
        """Return a fast, approximate frame for an index or timestamp."""

        if index is None and t_s is None:
            raise ValueError("Provide either index or t_s")
        if index is not None and t_s is not None:
            raise ValueError("Provide only one of index or t_s")

        if t_s is None:
            if index is None:
                raise ValueError("Provide either index or t_s")
            if index < 0:
                index += self.number_of_frames
            target_frame = int(index)
        else:
            t_s = float(t_s)
            target_frame = int(round(t_s * (self._nominal_frame_rate or self._frame_rate or 1.0)))

        if use_sequential:
            decoded = self._read_frame_fast_like(target_frame, decode_rgb=decode_rgb)
            self._last_fast_index = target_frame
            return decoded

        fps = self._nominal_frame_rate or self._frame_rate or 1.0
        target_pts = self._secs_to_pts(target_frame / fps)
        cached = self._frame_cache.get(target_pts)
        if cached is not None:
            self._last_fast_index = target_frame
            return cached  # type: ignore[return-value]

        decoded = self._read_frame_fast_opencv_pyav(target_frame, decode_rgb=decode_rgb)
        self._last_fast_index = target_frame
        return decoded

    def read_frame(
        self,
        *,
        index: Optional[int] = None,
        t_s: Optional[Number] = None,
        mode: str = "accurate",
        decode_rgb: bool = True,
        keyframe_mode: str = "nearest",
        use_sequential: bool = True,
    ) -> DecodedFrame:
        """Read a frame using a selectable access mode."""

        if mode not in {"accurate", "accurate_timeline", "fast", "scrub"}:
            raise ValueError("mode must be one of: 'accurate', 'accurate_timeline', 'fast', 'scrub'")
        if index is None and t_s is None:
            raise ValueError("Provide either index or t_s")
        if index is not None and t_s is not None:
            raise ValueError("Provide only one of index or t_s")

        if mode == "accurate":
            if index is not None:
                return self._read_frame_at_index(
                    int(index),
                    decode_rgb=decode_rgb,
                    use_sequential=use_sequential,
                )
            assert t_s is not None
            return self.read_frame_at(float(t_s), use_index=False)

        if mode == "accurate_timeline":
            if t_s is None:
                fps = self._nominal_frame_rate or self._frame_rate or 1.0
                t_s = float(index) / fps
            return self.read_frame_at(float(t_s))

        if mode == "scrub":
            if t_s is None:
                fps = self._frame_rate or 1.0
                t_s = float(index) / fps
            return self.read_keyframe_at(float(t_s), mode=keyframe_mode, decode_rgb=decode_rgb)

        return self.read_frame_fast(
            index=index,
            t_s=t_s,
            decode_rgb=decode_rgb,
            use_sequential=use_sequential,
        )

    def read_next_frame(self, *, decode_rgb: bool = True) -> DecodedFrame:
        """Return the next frame using sequential decoding."""

        self._ensure_seq_container()
        assert self._seq_decoder is not None
        try:
            frame = next(self._seq_decoder)
        except StopIteration:
            raise
        pts = frame.pts if frame.pts is not None else frame.dts
        img = frame.to_rgb().to_ndarray() if decode_rgb else frame.to_ndarray(format="bgr24")
        cur = DecodedFrame(
            image=img,
            pts=pts,
            time_s=self._frame_time_s(pts),
            key_frame=bool(getattr(frame, "key_frame", False)),
        )
        if pts is not None:
            self._frame_cache.put(int(pts), cur)
        self._current_frame_pos = float(self._seq_frame_index)
        self._last_index = int(self._seq_frame_index)
        self._last_fast_index = int(self._seq_frame_index)
        self._seq_frame_index += 1
        return cur

    def iter_frames(self, *, decode_rgb: bool = True) -> Iterator[DecodedFrame]:
        """Iterate through frames sequentially."""

        self.reset_sequence()
        assert self._seq_decoder is not None
        for frame in self._seq_decoder:
            pts = frame.pts if frame.pts is not None else frame.dts
            img = frame.to_rgb().to_ndarray() if decode_rgb else frame.to_ndarray(format="bgr24")
            cur = DecodedFrame(
                image=img,
                pts=pts,
                time_s=self._frame_time_s(pts),
                key_frame=bool(getattr(frame, "key_frame", False)),
            )
            if pts is not None:
                self._frame_cache.put(int(pts), cur)
            self._current_frame_pos = float(self._seq_frame_index)
            self._seq_frame_index += 1
            yield cur
