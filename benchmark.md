# Benchmarking

Use the benchmark suite to compare read modes, cache impact, and threading
tradeoffs on your own videos. The CLI prints timing tables and accuracy
statistics for every video it processes.

## Run on your own videos
```bash
# Single video
python -m acvr.benchmarks /path/to/video.mp4 --metric truth --compare-fastvideo

# All supported videos in a directory
python -m acvr.benchmarks /path/to/videos --metric truth --compare-fastvideo
```

The legacy wrapper `python -m scripts.benchmark_reader` is still supported.

## Reproduce bundled results
Run the benchmark on all bundled test assets:
```bash
python -m acvr.benchmarks tests/data --metric truth --compare-fastvideo \
  --index-pattern sequential --use-sequential on --samples 120 --no-opencv
```

Results (median ms/frame, local machine, PyAV 16.x):

| Video | Sequential | Accurate | Accurate TL | Scrub | Fast | FastVideo |
| --- | --- | --- | --- | --- | --- | --- |
| 10-03-22_Test 25_1-2v1-3.mp4 | 1.09 | 1.10 | 27.36 | 9.22 | 5.12 | 4.96 |
| 20190128_113421.mp4 | 1.37 | 1.37 | 37.13 | 37.51 | 1.44 | 1.39 |
| localhost-20181120_144618.mp4 | 0.32 | 0.32 | 114.95 | 79.19 | 61.06 | 57.71 |
| rpi9-20210409_093149.mp4 | 4.13 | 4.05 | 123.60 | 19.06 | 3.97 | 3.73 |
| test_cfr_h264.mp4 | 0.29 | 0.28 | 24.34 | 5.37 | 0.28 | 0.27 |
| test_vfr_h264.mp4 | 0.27 | 0.28 | 22.69 | 5.72 | 0.28 | 0.27 |

Sequential reads (`read_next` / iteration) and `use_sequential=True` provide
large speedups when accessing consecutive indices, bringing `accurate` and
`fast` close to true sequential decode performance.

## LRU cache impact
The decoded-frame LRU (`decoded_frame_cache_size`) helps when you revisit the
same indices repeatedly (e.g., UI scrubbing or repeated random samples). The
benchmark below repeats the same random index list three times to amplify cache
hits:

```bash
# No cache
python -m acvr.benchmarks tests/data/test_vfr_h264.mp4 --metric truth \
  --index-pattern random --use-sequential off --samples 200 --passes 3 \
  --cache-size 0 --no-opencv

# LRU cache enabled
python -m acvr.benchmarks tests/data/test_vfr_h264.mp4 --metric truth \
  --index-pattern random --use-sequential off --samples 200 --passes 3 \
  --cache-size 512 --no-opencv
```

Results (median ms/frame, local machine, PyAV 16.x):

| Mode | Cache=0 | Cache=512 |
| --- | --- | --- |
| Fast | 15.95 | 0.00 |
| Scrub | 6.08 | 0.01 |

Cached reads return immediately after the first pass, which is especially
useful for repeated random access patterns.

## Threading impact
Threading defaults to `on` and generally helps on H.264 assets, but some videos
may decode faster with threading disabled. The benchmark below uses random
indices and compares `on` vs `off`:

```bash
python -m acvr.benchmarks tests/data --metric truth --compare-fastvideo \
  --index-pattern random --use-sequential on --samples 40 --no-opencv --threading on

python -m acvr.benchmarks tests/data --metric truth --compare-fastvideo \
  --index-pattern random --use-sequential on --samples 40 --no-opencv --threading off
```

Results (median ms/frame, local machine, PyAV 16.x, `--samples 40`):

| Video | Accurate (on) | Accurate (off) | Fast (on) | Fast (off) |
| --- | --- | --- | --- | --- |
| 10-03-22_Test 25_1-2v1-3.mp4 | 21.06 | 28.18 | 5.44 | 12.47 |
| 20190128_113421.mp4 | 35.78 | 26.15 | 16.85 | 8.97 |
| localhost-20181120_144618.mp4 | 131.26 | 251.78 | 89.44 | 182.65 |
| rpi9-20210409_093149.mp4 | 103.46 | 212.26 | 28.57 | 123.90 |
| test_cfr_h264.mp4 | 39.25 | 93.62 | 12.08 | 54.36 |
| test_vfr_h264.mp4 | 31.48 | 77.59 | 12.07 | 50.65 |

## Flags
- `--index-pattern`: `sequential` (0..N-1) or `random` indices.
- `--use-sequential`: `on`, `off`, or `both` to control sequential switching.
- `--compare-fastvideo`: include a FastVideoReader-like baseline.
- `--threading`: enable or disable decoder threading.
- `--thread-count`: explicit FFmpeg decoder thread count (0 = auto).
- `--cache-size`: LRU decoded-frame cache size (0 disables).
- `--passes`: repeat the same index list multiple times.
- `--no-opencv`: omit OpenCV timing and accuracy rows.
- `--metric`: embedded|pts|truth (default truth).
- `--scrub-mode`: previous|nearest|next (default nearest).
- `--scrub-bucket-ms`: scrub cache bucket (default 25).
