import pytest

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


def test_read_next(tmp_path):
    """Validate sequential read_next access."""
    from acvr import VideoReader
    import numpy as np

    video_path = make_test_video(tmp_path)
    vr = VideoReader(video_path)

    for frame_number in range(5):
        frame = vr.read_next()
        brightness = np.mean(frame)
        assert brightness >= frame_number - 2 and brightness <= frame_number + 2


def test_read_frame_sequential_switch(tmp_path):
    """Ensure read_frame switches to sequential decoding."""
    from acvr import VideoReader
    import numpy as np

    video_path = make_test_video(tmp_path)
    vr = VideoReader(video_path)

    for frame_number in range(3):
        frame = vr.read_frame(index=frame_number, mode="accurate", use_sequential=True).image
        brightness = np.mean(frame)
        assert brightness >= frame_number - 2 and brightness <= frame_number + 2

    assert vr._backend._seq_decoder is not None
    assert vr._backend._seq_frame_index >= 3
