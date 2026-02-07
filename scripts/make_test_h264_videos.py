#!/usr/bin/env python3
import argparse
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def encode_frame_number(frame, frame_number):
    # Encode 32 bits into 2x2 blocks along the top row.
    for bit in range(32):
        value = (frame_number >> bit) & 1
        color = (0, 0, 255) if value else (0, 0, 0)
        x = 4 + bit * 2
        frame[0:2, x:x + 2] = color
    frame[0:2, 0:2] = (255, 255, 255)  # marker pixel block


def make_background(height, width, frame_number):
    x = np.linspace(0, 255, width, dtype=np.uint16)
    y = np.linspace(0, 255, height, dtype=np.uint16)
    xv, yv = np.meshgrid(x, y)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = ((xv + frame_number) % 256).astype(np.uint8)
    frame[:, :, 1] = ((yv + 2 * frame_number) % 256).astype(np.uint8)
    frame[:, :, 2] = (((xv // 2 + yv // 2) + 3 * frame_number) % 256).astype(np.uint8)
    return frame


def draw_text(frame, text):
    (height, _, _) = frame.shape
    origin = (20, height - 24)
    cv2.putText(frame, text, (origin[0] + 2, origin[1] + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def write_frames(out_dir, timestamps, width, height):
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, ts in enumerate(timestamps):
        frame = make_background(height, width, idx)
        encode_frame_number(frame, idx)
        text = f"frame {idx:05d}  t={ts:0.6f}s"
        draw_text(frame, text)
        cv2.imwrite(str(out_dir / f"{idx:06d}.png"), frame)


def ffconcat_escape(path):
    return str(path).replace("\\", "\\\\").replace("'", "\\'")


def write_concat_file(frame_dir, durations, concat_path):
    lines = []
    for idx, duration in enumerate(durations):
        frame_path = frame_dir / f"{idx:06d}.png"
        lines.append(f"file '{ffconcat_escape(frame_path)}'")
        if idx < len(durations) - 1:
            lines.append(f"duration {duration:0.6f}")
    concat_path.write_text("\n".join(lines) + "\n")


def run_ffmpeg(cmd):
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit("ffmpeg not found in PATH.")


def gop_args(mode):
    if mode == "hard":
        return ["-g", "240", "-keyint_min", "240", "-sc_threshold", "0",
                "-bf", "6", "-x264-params", "b-pyramid=strict:open_gop=1:ref=5"]
    if mode == "very-hard":
        return ["-g", "300", "-keyint_min", "300", "-sc_threshold", "0",
                "-bf", "8", "-x264-params", "b-pyramid=strict:open_gop=1:ref=8"]
    return ["-g", "120", "-keyint_min", "120", "-sc_threshold", "0", "-bf", "3"]


def main():
    parser = argparse.ArgumentParser(description="Generate H.264 CFR/VFR test videos.")
    parser.add_argument("--out-dir", default="tests/data", help="Output directory.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--cfr-fps", type=int, default=30)
    parser.add_argument("--gop-mode", choices=["default", "hard", "very-hard"],
                        default="hard",
                        help="GOP difficulty preset for random-access.")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_count = args.frames
    cfr_fps = args.cfr_fps

    cfr_timestamps = [i / cfr_fps for i in range(frame_count)]
    vfr_pattern = [1 / 24, 1 / 30, 1 / 15, 1 / 60, 1 / 30]
    vfr_durations = [vfr_pattern[i % len(vfr_pattern)] for i in range(frame_count)]
    vfr_timestamps = []
    current = 0.0
    for duration in vfr_durations:
        vfr_timestamps.append(current)
        current += duration

    temp_root = Path(tempfile.mkdtemp(prefix="h264_test_frames_"))
    cfr_frames = temp_root / "cfr"
    vfr_frames = temp_root / "vfr"

    write_frames(cfr_frames, cfr_timestamps, args.width, args.height)
    write_frames(vfr_frames, vfr_timestamps, args.width, args.height)

    cfr_output = out_dir / "test_cfr_h264.mp4"
    vfr_output = out_dir / "test_vfr_h264.mp4"
    noindex_output = out_dir / "test_noindex_h264.mp4"
    gop = gop_args(args.gop_mode)

    run_ffmpeg([
        "ffmpeg", "-y",
        "-framerate", str(cfr_fps),
        "-i", str(cfr_frames / "%06d.png"),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-crf", "18",
        *gop,
        "-movflags", "+faststart",
        str(cfr_output),
    ])

    concat_path = temp_root / "vfr_concat.txt"
    write_concat_file(vfr_frames, vfr_durations, concat_path)
    run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_path),
        "-vsync", "vfr",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-crf", "18",
        *gop,
        "-movflags", "+faststart",
        str(vfr_output),
    ])

    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(cfr_output),
        "-c", "copy",
        "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
        str(noindex_output),
    ])

    if args.keep_frames:
        shutil.copytree(cfr_frames, out_dir / "frames_cfr", dirs_exist_ok=True)
        shutil.copytree(vfr_frames, out_dir / "frames_vfr", dirs_exist_ok=True)

    shutil.rmtree(temp_root, ignore_errors=True)
    print(f"Wrote {cfr_output}")
    print(f"Wrote {vfr_output}")
    print(f"Wrote {noindex_output}")


if __name__ == "__main__":
    main()
