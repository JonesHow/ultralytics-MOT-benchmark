# Ultralytics MOT Benchmark

## Quickstart

- Run inference (CLI):
  - Install deps, then run: `python -m pip install -e .`
  - Use the console script: `mot-infer --weights models/yolo12x.pt --source data/samples/camera-d8accaf0_10s.mp4 --output outputs/videos`
  - Overrides example: `mot-infer --track-overrides "conf=0.45,imgsz=800,classes=[0]"`

- Paths
  - Configs JSON: `configs/track_configs.json` (override with `--track-configs-path`)
  - Outputs: `outputs/videos` (MP4 + matching JSON metadata)
  - Logs: `logs/`

- Testing
  - `pytest -q` (tests import from `src/` layout)

## Docker (dev)

- Start environment:
  - `docker compose up -d`
  - Container mounts the repo at `/app`, installs deps via `uv pip sync`, then `pip install -e .` for the CLI.
  - Defaults: `TRACK_CONFIGS_PATH=configs/track_configs.json`, `OUTPUT_DIR=outputs/videos`.

- Run in container:
  - `docker compose exec dev-env mot-infer --weights models/yolo12x.pt --source data/samples/camera-d8accaf0_10s.mp4`
  - Or pass overrides: `--track-overrides "conf=0.45,imgsz=800,classes=[0]"`

## Parameter Testing

- Run parameter tests:
  - `mot-params --config configs/test_config.yaml --output-dir outputs/reports`
  - Analyze existing results only: `mot-params --analyze-only --results-file outputs/reports/parameter_test_report_....json --output-dir outputs/reports`

- Config fields under `test_settings` used by the tester:
  - `video_source` (e.g., `data/samples/camera-d8accaf0_60s.mp4`)
  - `weights` (e.g., `models/yolo12x.pt`)
  - Optional: `output_dir` (default: `outputs/videos`)
  - Optional: `track_config_name` (default: `default`)
  - Optional: `track_configs_path` (default: `configs/track_configs.json`)

## OpenCV ImportError: libGL.so.1

If you see an error like:

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

This is caused by `opencv-python` depending on system GUI libraries (libGL). This
project is headless and does not require GUI windows, so switch to the headless
build of OpenCV.

Suggested fix:

- Use the headless wheel (preferred):
  - Update `pyproject.toml` to use `opencv-python-headless`.
  - Then reinstall dependencies:
    - With Docker Compose: the service runs `uv pip sync --system pyproject.toml`, just restart/rebuild the container.
    - Locally:
      - `pip uninstall -y opencv-python opencv-contrib-python`
      - `pip install --upgrade opencv-python-headless`

- Alternatively, install OS libraries to satisfy `opencv-python`:
  - Debian/Ubuntu: `apt-get update && apt-get install -y libgl1 libglib2.0-0`
  - Also sometimes needed: `libxext6 libsm6 libxrender1`

Note: The scripts default `view_img=False`, so headless OpenCV works out of the box.
