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
            })


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

    return {
        "json": json_path,
        "csv": csv_path,
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
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

