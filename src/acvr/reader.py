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
        build_index: bool = False,
        decoded_frame_cache_size: int = 0,
        scrub_bucket_ms: int = 100,
        scrub_bucket_lru_size: int = 4096,
    ) -> None:
        """Create a reader for the given video path."""

        self._backend = PyAVVideoBackend(
            path,
            video_stream_index=video_stream_index,
            build_index=build_index,
            decoded_frame_cache_size=decoded_frame_cache_size,
            scrub_bucket_ms=scrub_bucket_ms,
            scrub_bucket_lru_size=scrub_bucket_lru_size,
        )

    def close(self) -> None:
        """Close the underlying video resources."""

        self._backend.close()

    def __enter__(self) -> "VideoReader":
        """Return self for context manager usage."""

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the reader when leaving a context manager."""

        self.close()

    def __len__(self) -> int:
        """Return the number of frames in the video."""

        return self.number_of_frames

    def __getitem__(self, key: IndexKey) -> Union[np.ndarray, List[np.ndarray]]:
        """Return a frame or list of frames for the given index or slice."""

        if isinstance(key, slice):
            start, stop, step = key.indices(self.number_of_frames)
            return [self._backend.frame_at_index(i) for i in range(start, stop, step)]
        return self._backend.frame_at_index(int(key))

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over all frames in the video."""

        return iter(self[:])

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
        """Build a keyframe index for faster random access."""

        return self._backend.build_keyframe_index(max_packets=max_packets)

    def read_keyframe_at(
        self,
        t_s: float,
        *,
        mode: str = "previous",
        decode_rgb: bool = True,
    ) -> DecodedFrame:
        """Return a nearby keyframe for fast scrubbing."""

        return self._backend.read_keyframe_at(t_s, mode=mode, decode_rgb=decode_rgb)

    def read_frame_at(
        self,
        t_s: float,
        *,
        return_first_after: bool = True,
        max_decode_frames: int = 10_000,
        use_index: bool = True,
    ) -> DecodedFrame:
        """Return a frame at a timestamp with accurate seeking."""

        return self._backend.read_frame_at(
            t_s,
            return_first_after=return_first_after,
            max_decode_frames=max_decode_frames,
            use_index=use_index,
        )
