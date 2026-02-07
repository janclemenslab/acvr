"""User-facing video reader interface."""

from __future__ import annotations

from typing import Iterator, List, Optional, Union

import numpy as np

from acvr._pyav_backend import DecodedFrame, KeyframeEntry, PyAVVideoBackend

IndexKey = Union[int, slice]


class VideoReader:
    """High-level video reader with array-style access."""

    def __init__(
        self,
        path: str,
        video_stream_index: int = 0,
        *,
        build_index: bool = True,
        decoded_frame_cache_size: int = 0,
        scrub_bucket_ms: int = 25,
        scrub_bucket_lru_size: int = 4096,
        threading: bool = True,
        thread_count: int = 0,
        index_policy: str = "decode",
    ) -> None:
        """Create a reader for the given video path.

        Args:
            path: Path to the video file to open.
            video_stream_index: Video stream index to decode.
            build_index: Whether to build a keyframe index on initialization (default
                True); can speed up accurate random seeks but adds upfront cost.
                speed up accurate random seeks but adds upfront cost.
            decoded_frame_cache_size: Number of decoded frames to keep in an
                in-memory LRU cache; helpful for repeated access to nearby frames.
            scrub_bucket_ms: Bucket size (milliseconds) used to group timestamps
                for fast scrub queries.
            scrub_bucket_lru_size: LRU size for the scrub bucket cache.
            threading: Whether to enable threaded decoding in the backend.
            thread_count: Number of decoding threads (0 lets backend decide).
            index_policy: Indexing policy, either ``"decode"`` for decode-order
                frames or ``"timeline"`` for timestamp-based access.

        Raises:
            ValueError: If ``index_policy`` is not ``"decode"`` or ``"timeline"``.
        """

        self._backend = PyAVVideoBackend(
            path,
            video_stream_index=video_stream_index,
            build_index=build_index,
            decoded_frame_cache_size=decoded_frame_cache_size,
            scrub_bucket_ms=scrub_bucket_ms,
            scrub_bucket_lru_size=scrub_bucket_lru_size,
            threading=threading,
            thread_count=thread_count,
        )
        if index_policy not in {"decode", "timeline"}:
            raise ValueError("index_policy must be 'decode' or 'timeline'")
        self._index_policy = index_policy

    def close(self) -> None:
        """Close the underlying video resources."""

        self._backend.close()

    def __enter__(self) -> "VideoReader":
        """Return self for context manager usage."""

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the reader when leaving a context manager.

        Args:
            exc_type: Exception type, if any.
            exc: Exception instance, if any.
            tb: Traceback, if any.
        """

        self.close()

    def __len__(self) -> int:
        """Return the number of frames in the video."""

        return self.number_of_frames

    def __getitem__(self, key: IndexKey) -> Union[np.ndarray, List[np.ndarray]]:
        """Return a frame or list of frames for the given index or slice.

        Indexing semantics are controlled by `index_policy`:
        - 'decode' (default): index refers to decode-order frame number.
        - 'timeline': index is mapped to timestamp using nominal FPS, and an accurate
          timestamp seek is performed.

        Args:
            key: Frame index or slice to retrieve.

        Returns:
            A single frame array or list of frame arrays.
        """

        if isinstance(key, slice):
            start, stop, step = key.indices(self.number_of_frames)
            return [self[i] for i in range(start, stop, step)]

        i = int(key)
        if self._index_policy == "decode":
            return self._backend.frame_at_index(i)
        # timeline policy
        fps = self.nominal_frame_rate or self.frame_rate or 1.0
        t_s = float(i) / fps
        return self._backend.read_frame_at(t_s).image

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over all frames in the video."""

        return self.iter_frames()

    @property
    def frame_height(self) -> int:
        """Return the frame height in pixels."""

        return self._backend.frame_height

    @property
    def frame_width(self) -> int:
        """Return the frame width in pixels."""

        return self._backend.frame_width

    @property
    def frame_rate(self) -> float:
        """Return the video frame rate."""

        return self._backend.frame_rate

    @property
    def nominal_frame_rate(self) -> float:
        """Return the nominal video frame rate (guessed_rate when available)."""

        return self._backend.nominal_frame_rate

    @property
    def fourcc(self) -> int:
        """Return the fourcc codec identifier."""

        return self._backend.fourcc

    @property
    def frame_format(self) -> int:
        """Return the pixel format identifier."""

        return self._backend.frame_format

    @property
    def number_of_frames(self) -> int:
        """Return the total number of frames."""

        return self._backend.number_of_frames

    @property
    def frame_shape(self) -> tuple:
        """Return the expected frame shape (H, W, C)."""

        return self._backend.frame_shape

    @property
    def current_frame_pos(self) -> float:
        """Return the last accessed frame index."""

        return self._backend.current_frame_pos

    def build_keyframe_index(self, *, max_packets: Optional[int] = None) -> List[KeyframeEntry]:
        """Build a keyframe index for faster random access.

        Args:
            max_packets: Optional cap on packets to inspect.

        Returns:
            A list of keyframe entries.
        """

        return self._backend.build_keyframe_index(max_packets=max_packets)

    def read_keyframe_at(
        self,
        t_s: float,
        *,
        mode: str = "nearest",
        decode_rgb: bool = True,
    ) -> DecodedFrame:
        """Return a nearby keyframe for fast scrubbing.

        Args:
            t_s: Timestamp in seconds to seek around.
            mode: Selection mode (``"nearest"``, ``"before"``, or ``"after"``).
            decode_rgb: Whether to decode into RGB arrays.

        Returns:
            The decoded keyframe.
        """

        return self._backend.read_keyframe_at(t_s, mode=mode, decode_rgb=decode_rgb)

    def read_frame_at(
        self,
        t_s: float,
        *,
        return_first_after: bool = True,
        max_decode_frames: int = 10_000,
        use_index: bool = True,
    ) -> DecodedFrame:
        """Return a frame at a timestamp with accurate seeking.

        Args:
            t_s: Timestamp in seconds to seek to.
            return_first_after: Return the first frame after the timestamp.
            max_decode_frames: Cap on frames to decode while seeking.
            use_index: Whether to use the keyframe index if available.

        Returns:
            The decoded frame at the target timestamp.
        """

        return self._backend.read_frame_at(
            t_s,
            return_first_after=return_first_after,
            max_decode_frames=max_decode_frames,
            use_index=use_index,
        )

    def read_frame_fast(
        self,
        *,
        index: Optional[int] = None,
        t_s: Optional[float] = None,
        decode_rgb: bool = True,
        use_sequential: bool = True,
    ) -> DecodedFrame:
        """Return a fast, approximate frame for an index or timestamp.

        Args:
            index: Decode-order frame index to seek to.
            t_s: Timestamp in seconds to seek to.
            decode_rgb: Whether to decode into RGB arrays.
            use_sequential: Allow sequential decoding when available.

        Returns:
            The decoded frame closest to the request.
        """

        return self._backend.read_frame_fast(
            index=index,
            t_s=t_s,
            decode_rgb=decode_rgb,
            use_sequential=use_sequential,
        )

    def read_next(self, *, decode_rgb: bool = True) -> np.ndarray:
        """Return the next frame using sequential decoding.

        Args:
            decode_rgb: Whether to decode into RGB arrays.

        Returns:
            The next decoded frame image.
        """

        return self._backend.read_next_frame(decode_rgb=decode_rgb).image

    def iter_frames(self, *, decode_rgb: bool = True) -> Iterator[np.ndarray]:
        """Iterate frames sequentially without seeking."""

        for frame in self._backend.iter_frames(decode_rgb=decode_rgb):
            yield frame.image

    # Public PTS/time helpers
    def pts_at_index(self, index: int) -> Optional[int]:
        """Return the PTS for a given frame index."""

        return self._backend.pts_at_index(int(index))

    def time_at_index(self, index: int) -> float:
        """Return the timestamp (seconds) for a given frame index."""

        return self._backend.time_at_index(int(index))

    def index_from_pts(self, pts: int) -> int:
        """Return the nearest frame index for a PTS value."""

        return self._backend.index_from_pts(int(pts))

    def index_from_time(self, t_s: float) -> int:
        """Return the nearest frame index for a timestamp in seconds."""

        return self._backend.index_from_time(float(t_s))

    def read_frame(
        self,
        *,
        index: Optional[int] = None,
        t_s: Optional[float] = None,
        mode: str = "accurate",
        decode_rgb: bool = True,
        keyframe_mode: str = "nearest",
        use_sequential: bool = True,
    ) -> DecodedFrame:
        """Read a frame using a selectable access mode."""

        return self._backend.read_frame(
            index=index,
            t_s=t_s,
            mode=mode,
            decode_rgb=decode_rgb,
            keyframe_mode=keyframe_mode,
            use_sequential=use_sequential,
        )
