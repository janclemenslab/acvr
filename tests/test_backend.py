"""Tests for the PyAV backend implementation."""

from pathlib import Path

import numpy as np
import pytest

from acvr._pyav_backend import PyAVVideoBackend


def make_test_video(path: Path) -> str:
    """Create a simple MJPG test video."""

    cv2 = pytest.importorskip("cv2")
    video_path = str(path / "test_backend.avi")
    vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48), False)
    for frame_number in range(20):
        frame = (np.ones((48, 64)) * frame_number).astype("uint8")
        vw.write(frame)
    vw.release()
    return video_path


def test_backend_frame_at_index(tmp_path: Path) -> None:
    """Ensure backend can decode by index."""

    video_path = make_test_video(tmp_path)
    backend = PyAVVideoBackend(video_path)
    try:
        frame = backend.frame_at_index(5)
        assert frame.shape[:2] == (48, 64)
        assert 4 <= frame.mean() <= 6
    finally:
        backend.close()


def test_backend_keyframe_index() -> None:
    """Ensure keyframe index is built and ordered."""

    video_path = Path(__file__).resolve().parent / "data" / "test_cfr_h264.mp4"
    if not video_path.exists():
        pytest.skip(f"Missing test video: {video_path}")

    backend = PyAVVideoBackend(str(video_path))
    try:
        keyframes = backend.build_keyframe_index()
        assert keyframes
        pts_list = [entry.pts for entry in keyframes]
        assert pts_list == sorted(pts_list)
        times = [entry.time_s for entry in keyframes]
        assert times == sorted(times)
    finally:
        backend.close()


def test_backend_read_frame_at() -> None:
    """Ensure timestamp-based read returns a decoded frame."""

    video_path = Path(__file__).resolve().parent / "data" / "test_cfr_h264.mp4"
    if not video_path.exists():
        pytest.skip(f"Missing test video: {video_path}")

    backend = PyAVVideoBackend(str(video_path))
    try:
        frame = backend.read_frame_at(0.0)
        assert frame.image.ndim == 3
        assert frame.time_s >= 0.0
    finally:
        backend.close()


def test_backend_read_keyframe_at() -> None:
    """Ensure keyframe read returns a decoded frame."""

    video_path = Path(__file__).resolve().parent / "data" / "test_cfr_h264.mp4"
    if not video_path.exists():
        pytest.skip(f"Missing test video: {video_path}")

    backend = PyAVVideoBackend(str(video_path), build_index=True)
    try:
        frame = backend.read_keyframe_at(0.5)
        assert frame.image.ndim == 3
        assert frame.pts is not None
    finally:
        backend.close()


def test_backend_read_keyframe_at_edges() -> None:
    """Keyframe seeks should work at start and near end of stream."""

    video_path = Path(__file__).resolve().parent / "data" / "test_cfr_h264.mp4"
    if not video_path.exists():
        pytest.skip(f"Missing test video: {video_path}")

    backend = PyAVVideoBackend(str(video_path), build_index=True)
    try:
        # Start of stream
        f0 = backend.read_keyframe_at(0.0)
        assert f0.image.ndim == 3
        assert f0.key_frame

        # Near end of stream
        fps = backend.frame_rate or 1.0
        t_end = (backend.number_of_frames - 1) / fps
        f_end = backend.read_keyframe_at(max(0.0, t_end - 1e-3))
        assert f_end.image.ndim == 3
        assert f_end.key_frame
    finally:
        backend.close()

def test_backend_read_frame_fast() -> None:
    """Ensure fast frame read returns a decoded frame."""

    video_path = Path(__file__).resolve().parent / "data" / "test_cfr_h264.mp4"
    if not video_path.exists():
        pytest.skip(f"Missing test video: {video_path}")

    backend = PyAVVideoBackend(str(video_path))
    try:
        frame = backend.read_frame_fast(index=10, decode_rgb=True)
        assert frame.image.ndim == 3
        assert frame.pts is not None
        with pytest.raises(ValueError):
            backend.read_frame_fast(index=1, t_s=0.1)
    finally:
        backend.close()
