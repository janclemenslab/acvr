import pytest
from pathlib import Path
from functools import lru_cache

import numpy as np


def make_test_video(path):
    """Generate a small MJPG test video on disk."""
    import cv2
    import numpy as np

    video_path = str(path / "test.avi")

    vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 480), False)
    for frame_number in range(255):
        frame = (np.ones((480, 640)) * frame_number).astype("uint8")
        print(np.mean(frame))
        vw.write(frame)

    return video_path


def test_import():
    """Ensure the public reader can be imported."""

    from acvr import VideoReader


def test_frame_attrs(tmp_path):
    """Validate basic metadata properties."""
    import cv2
    from acvr import VideoReader

    video_path = make_test_video(tmp_path)
    vr = VideoReader(video_path)

    assert vr.frame_height == 480
    assert vr.frame_width == 640
    assert vr.frame_rate == 25.0
    assert vr.fourcc == cv2.VideoWriter_fourcc(*"MJPG")
    assert vr.frame_format == 0
    assert vr.number_of_frames == 255
    assert vr.frame_shape == (480, 640, 3)
    assert vr.current_frame_pos == 0.0


def test_index(tmp_path):
    """Validate random access by frame index."""
    from acvr import VideoReader
    import numpy as np

    video_path = make_test_video(tmp_path)
    vr = VideoReader(video_path)

    for frame_number in range(vr.number_of_frames):
        frame = vr[frame_number]
        brightness = np.mean(frame)
        assert brightness >= frame_number - 2 and brightness <= frame_number + 2


def test_iter(tmp_path):
    """Validate iteration over all frames."""
    from acvr import VideoReader
    import numpy as np

    video_path = make_test_video(tmp_path)
    vr = VideoReader(video_path)

    for frame_number, frame in enumerate(vr[:]):
        brightness = np.mean(frame)
        assert brightness >= frame_number - 2 and brightness <= frame_number + 2


def test_slice(tmp_path):
    """Validate slice-based access."""
    from acvr import VideoReader
    import numpy as np

    video_path = make_test_video(tmp_path)
    vr = VideoReader(video_path)

    step = 10
    for index, frame in enumerate(vr[::step]):
        frame_number = index * step
        brightness = np.mean(frame)
        assert brightness >= frame_number - 2 and brightness <= frame_number + 2


def decode_frame_number(frame, channel_index):
    """Decode the embedded frame index in the test fixtures."""
    bits = []
    for bit in range(32):
        x0 = 4 + bit * 2
        block = frame[0:2, x0 : x0 + 2, channel_index]
        bits.append(1 if block.mean() > 127 else 0)
    value = 0
    for bit, flag in enumerate(bits):
        value |= flag << bit
    return value


@lru_cache(maxsize=None)
def expected_frame_numbers(video_path_str: str):
    """Return expected frame indices from PyAV decoding."""
    av = pytest.importorskip("av")
    container = av.open(video_path_str)
    stream = container.streams.video[0]
    numbers = []
    for frame in container.decode(stream):
        numbers.append(decode_frame_number(frame.to_rgb().to_ndarray(), channel_index=0))
    container.close()
    return numbers


def assert_random_access_opencv_vs_acvr(video_path):
    """Compare OpenCV random access to acvr output."""
    if not video_path.exists():
        pytest.skip(f"Missing test video: {video_path}")

    cv2 = pytest.importorskip("cv2")
    from acvr import VideoReader

    with VideoReader(str(video_path)) as vr:
        frame_count = vr.number_of_frames

    expected_numbers = expected_frame_numbers(str(video_path))
    if expected_numbers:
        frame_count = min(frame_count, len(expected_numbers))

    if frame_count <= 0:
        pytest.skip(f"Video reports no frames: {video_path}")

    indices = np.linspace(0, frame_count - 1, num=min(50, frame_count), dtype=int)
    indices = sorted(set(indices.tolist()))

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()

    mismatches = 0
    checked = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        checked += 1
        if not ok or frame is None:
            mismatches += 1
            continue
        decoded = decode_frame_number(frame, channel_index=2)
        if decoded != idx:
            mismatches += 1
    cap.release()

    assert checked > 0

    with VideoReader(str(video_path)) as vr:
        for idx in indices:
            frame = vr[idx]
            decoded = decode_frame_number(frame, channel_index=0)
            assert decoded == expected_numbers[idx]


def test_random_access_cfr_opencv_vs_acvr():
    """Validate random access for CFR test asset."""
    video_path = Path(__file__).resolve().parent / "data" / "test_cfr_h264.mp4"
    assert_random_access_opencv_vs_acvr(video_path)


def test_random_access_vfr_opencv_vs_acvr():
    """Validate random access for VFR test asset."""
    video_path = Path(__file__).resolve().parent / "data" / "test_vfr_h264.mp4"
    assert_random_access_opencv_vs_acvr(video_path)
