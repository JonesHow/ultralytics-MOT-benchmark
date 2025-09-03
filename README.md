# Ultralytics MOT Benchmark

## Quickstart

- Run inference (CLI):
  - Install deps, then run: `python -m pip install -e .`
  - Use the console script: `uv run mot-infer --weights models/yolo12x.pt --source data/samples/camera-d8accaf0_10s.mp4 --output outputs/videos`
  - Overrides example: `uv run mot-infer --track-overrides "conf=0.45,imgsz=800,classes=[0]"`

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
  - `docker compose exec dev-env uv run mot-infer --weights models/yolo12x.pt --source data/samples/camera-d8accaf0_10s.mp4`
  - Or pass overrides: `docker compose exec dev-env uv run mot-infer --track-overrides "conf=0.45,imgsz=800,classes=[0]"`

## Parameter Testing

- Run parameter tests:
  - `uv run mot-params --config configs/test_config.yaml --output-dir outputs/reports`
  - Analyze existing results only: `uv run mot-params --analyze-only --results-file outputs/reports/parameter_test_report_....json --output-dir outputs/reports`

- Config fields under `test_settings` used by the tester:
  - `video_source` (e.g., `data/samples/camera-d8accaf0_60s.mp4`)
  - `weights` (e.g., `models/yolo12x.pt`)
  - Optional: `output_dir` (default: `outputs/videos`)
  - Optional: `track_config_name` (default: `default`)
  - Optional: `track_configs_path` (default: `configs/track_configs.json`)

## Analyze JSON Results

The analyzer needs only the JSON metadata files produced by inference. Point it at any folder that contains those JSONs (for example `outputs/videos/`), and it will rank runs by tracking quality and generate reports. Log files in `logs/` are not required for this analyzer.

- Run (console script):
  - Analyze curated JSON folder: `uv run mot-analyze --dir .temp/filter_logs --out-dir outputs/reports --top-k 5`
  - Analyze JSONs directly under `outputs/videos/`: `uv run mot-analyze --dir outputs/videos --pattern "*.json" --out-dir outputs/reports`
  - Analyze a subset by name pattern: `uv run mot-analyze --dir outputs/videos --pattern "camera-*.json" --out-dir outputs/reports`

- Run with uv (without installing scripts):
  - `uv run -m ultralytics_mot_benchmark.cli.analyze_filter_logs --dir outputs/videos --pattern "*.json" --out-dir outputs/reports --top-k 5`

- Options:
  - `--dir`: input folder containing JSON reports (default: `.temp/filter_logs`)
  - `--pattern`: glob pattern to select JSONs (default: `*.json`)
  - `--out-dir`: output folder for reports (default: `outputs/reports`)
  - `--top-k`: number of top results to summarize (default: `10`)

- Outputs (saved under `--out-dir`):
  - Ranking JSON: `filter_logs_ranking_<timestamp>.json`
  - Ranking CSV: `filter_logs_ranking_<timestamp>.csv`
    - Includes a `parameters` column (JSON string) for quick inspection
  - Top‑K parameters JSON: `filter_logs_topk_parameters_<timestamp>.json`
  - HTML report: `filter_logs_report_<timestamp>.html`
  - Figures directory: `filter_logs_figures_<timestamp>/` containing PNG charts

- Charts included in the HTML report:
  - Top‑K Overall Score bar chart
  - Quality component bars: continuity/fragmentation/efficiency/stability
  - Avg FPS vs Overall Score scatter
  - Parameter variation bar (unique values across Top‑K, top 20)
  - Numeric parameter heatmap (min‑max normalized)
  - Categorical parameter heatmap (encoded to 0–1)

Notes
- Only numeric indicators are plotted as metrics; non‑numeric indicators are skipped.
- CSV `parameters` is a JSON string for downstream processing.
- Figures are referenced in the HTML via relative paths placed in the figures directory.

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
