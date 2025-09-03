#!/usr/bin/env python3
"""
掃描 .temp/filter_logs 的 JSON 結果，
用追蹤品質分析器進行評分與排名，輸出 JSON 與 CSV 摘要。
"""

import os
import glob
import json
import csv
import argparse
from datetime import datetime
from typing import List, Dict, Any

from ultralytics_mot_benchmark.analysis.tracking_quality_analyzer import (
    compare_multiple_results,
    analyze_tracking_result,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _collect_jsons(input_dir: str, pattern: str) -> List[str]:
    search_glob = os.path.join(input_dir, pattern)
    return sorted(glob.glob(search_glob))


def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_csv(path: str, ranking: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "file",
        "run_id",
        "overall_score",
        "continuity_score",
        "fragmentation_score",
        "efficiency_score",
        "stability_score",
        "avg_fps",
        "continuity_ratio",
        "fragmentation_ratio",
        "id_switches",
        "total_tracks",
        "avg_track_length",
        "parameters",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, r in enumerate(ranking, 1):
            qs = r.get("quality_scores", {})
            km = r.get("key_metrics", {})
            writer.writerow({
                "rank": idx,
                "file": r.get("file_path", ""),
                "run_id": r.get("run_id", ""),
                "overall_score": qs.get("overall_score", ""),
                "continuity_score": qs.get("continuity_score", ""),
                "fragmentation_score": qs.get("fragmentation_score", ""),
                "efficiency_score": qs.get("efficiency_score", ""),
                "stability_score": qs.get("stability_score", ""),
                "avg_fps": km.get("avg_fps", ""),
                "continuity_ratio": km.get("continuity_ratio", ""),
                "fragmentation_ratio": km.get("fragmentation_ratio", ""),
                "id_switches": km.get("id_switches", ""),
                "total_tracks": km.get("total_tracks", ""),
                "avg_track_length": km.get("avg_track_length", ""),
                "parameters": json.dumps(r.get("parameters", {}), ensure_ascii=False),
            })


def _write_topk_json(path: str, top_list: List[Dict[str, Any]]) -> None:
    out = []
    for idx, r in enumerate(top_list, 1):
        out.append({
            "rank": idx,
            "file": r.get("file_path"),
            "run_id": r.get("run_id"),
            "quality_scores": r.get("quality_scores", {}),
            "key_metrics": r.get("key_metrics", {}),
            "parameters": r.get("parameters", {}),
        })
    _write_json(path, out)


def _write_html(path: str, timestamp: str, source_dir: str, total: int,
                best: Dict[str, Any], top_list: List[Dict[str, Any]]) -> None:
    def esc(x: Any) -> str:
        try:
            return (str(x) if x is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        except Exception:
            return ""

    rows_html = []
    for idx, r in enumerate(top_list, 1):
        qs = r.get("quality_scores", {})
        km = r.get("key_metrics", {})
        rows_html.append(
            f"<tr>"
            f"<td>{idx}</td>"
            f"<td>{esc(r.get('run_id'))}</td>"
            f"<td>{esc(r.get('file_path'))}</td>"
            f"<td>{esc(qs.get('overall_score'))}</td>"
            f"<td>{esc(km.get('avg_fps'))}</td>"
            f"<td>{esc(km.get('continuity_ratio'))}</td>"
            f"<td>{esc(km.get('fragmentation_ratio'))}</td>"
            f"<td>{esc(km.get('id_switches'))}</td>"
            f"</tr>"
        )

    params_sections = []
    for idx, r in enumerate(top_list, 1):
        params = r.get("parameters", {}) or {}
        kv_rows = "".join(
            f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in sorted(params.items())
        ) or "<tr><td colspan=2><em>No parameters</em></td></tr>"
        params_sections.append(
            f"""
            <section class=card>
              <h3>#{idx} {esc(r.get('run_id'))}</h3>
              <table class=kv>
                <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
                <tbody>
                  {kv_rows}
                </tbody>
              </table>
            </section>
            """
        )

    best_fp = esc(best.get("file")) if best else ""
    best_id = esc(best.get("run_id")) if best else ""
    best_score = esc(best.get("overall_score")) if best else ""

    html = f"""
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Filter Logs Analysis Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #2c3e50; }}
    h1 {{ margin: 0 0 8px; }}
    .muted {{ color: #7f8c8d; }}
    .summary {{ background: #ecf0f1; padding: 16px; border-radius: 8px; margin: 16px 0 24px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f7f7f7; }}
    .card {{ border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 16px 0; }}
    .kv th, .kv td {{ border: 1px solid #eee; }}
  </style>
  </head>
<body>
  <h1>Filter Logs Analysis Report</h1>
  <div class="muted">Generated at {esc(timestamp)} · Source: {esc(source_dir)} · Total results: {total}</div>

  <div class="summary">
    <strong>Best:</strong>
    <div>Run ID: {best_id}</div>
    <div>File: {best_fp}</div>
    <div>Overall Score: {best_score}</div>
  </div>

  <h2>Top Results</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Run ID</th>
        <th>File</th>
        <th>Overall Score</th>
        <th>Avg FPS</th>
        <th>Continuity Ratio</th>
        <th>Fragmentation Ratio</th>
        <th>ID Switches</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>

  <h2>Top-K Parameters</h2>
  {''.join(params_sections)}

  <footer class="muted">This report is auto-generated.</footer>
</body>
</html>
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def run(input_dir: str, out_dir: str, pattern: str, top_k: int) -> Dict[str, Any]:
    paths = _collect_jsons(input_dir, pattern)
    if not paths:
        return {"error": f"No JSON files found under {input_dir} with pattern {pattern}"}

    summary = compare_multiple_results(paths)
    if "error" in summary:
        return summary

    # 產出資料
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _ensure_dir(out_dir)

    ranking = summary.get("ranking", [])
    top_list = ranking[: max(1, top_k)]

    # JSON 摘要
    json_out = {
        "generated_at": timestamp,
        "source_dir": os.path.abspath(input_dir),
        "total_results": summary.get("total_results", 0),
        "best": {
            "file": summary.get("best_result", {}).get("file_path"),
            "run_id": summary.get("best_result", {}).get("run_id"),
            "overall_score": summary.get("best_result", {}).get("quality_scores", {}).get("overall_score"),
            "parameters": summary.get("best_result", {}).get("parameters", {}),
            "key_metrics": summary.get("best_result", {}).get("key_metrics", {}),
        },
        "average_scores": summary.get("average_scores", {}),
        "top_k": top_k,
        "ranking": top_list,
    }

    json_path = os.path.join(out_dir, f"filter_logs_ranking_{timestamp}.json")
    _write_json(json_path, json_out)

    # CSV 排名
    csv_path = os.path.join(out_dir, f"filter_logs_ranking_{timestamp}.csv")
    _write_csv(csv_path, ranking)

    # Top-K 參數明細 JSON
    topk_params_path = os.path.join(out_dir, f"filter_logs_topk_parameters_{timestamp}.json")
    _write_topk_json(topk_params_path, top_list)

    # HTML 報告
    html_path = os.path.join(out_dir, f"filter_logs_report_{timestamp}.html")
    _write_html(html_path, timestamp, os.path.abspath(input_dir), summary.get("total_results", 0), json_out.get("best", {}), top_list)

    return {
        "json": json_path,
        "csv": csv_path,
        "html": html_path,
        "topk_json": topk_params_path,
        "total": summary.get("total_results", 0),
        "best_file": json_out["best"].get("file"),
        "best_score": json_out["best"].get("overall_score"),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze filtered JSON logs and rank results")
    parser.add_argument("--dir", default=os.getenv("FILTER_LOGS_DIR", ".temp/filter_logs"), help="input directory")
    parser.add_argument("--pattern", default=os.getenv("FILTER_LOGS_PATTERN", "*.json"), help="glob pattern")
    parser.add_argument("--out-dir", default=os.getenv("OUTPUT_DIR", "outputs/reports"), help="output directory")
    parser.add_argument("--top-k", type=int, default=10, help="number of top results to summarize")

    args = parser.parse_args()

    try:
        result = run(args.dir, args.out_dir, args.pattern, args.top_k)
        if "error" in result:
            print(f"Error: {result['error']}")
            raise SystemExit(1)

        print("=== Filter Logs Analysis ===")
        print(f"Input dir: {os.path.abspath(args.dir)}")
        print(f"Total results: {result['total']}")
        print(f"Best file: {result['best_file']}")
        print(f"Best score: {result['best_score']}")
        print(f"JSON: {result['json']}")
        print(f"CSV:  {result['csv']}")
        print(f"HTML: {result['html']}")
        print(f"TopK: {result['topk_json']}")
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
