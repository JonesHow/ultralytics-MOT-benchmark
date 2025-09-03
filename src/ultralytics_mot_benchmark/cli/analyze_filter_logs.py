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
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def _write_html_with_charts(path: str, timestamp: str, source_dir: str, total: int,
                            best: Dict[str, Any], top_list: List[Dict[str, Any]],
                            charts: Dict[str, str]) -> None:
    # 先生成基本 HTML 內容
    # 重用 _write_html 的內容，並在中間插入圖表區塊
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

    # 圖表區塊 HTML
    chart_html_parts = []
    def add_fig(title: str, key: str, width: int = 1000):
        src = charts.get(key)
        if not src:
            return
        chart_html_parts.append(
            f"<figure class=card>\n  <figcaption><strong>{esc(title)}</strong></figcaption>\n  <img src=\"{esc(src)}\" alt=\"{esc(title)}\" style=\"max-width:100%; border:1px solid #eee\"/>\n</figure>"
        )

    add_fig("Top-K Overall Score", "topk_overall")
    add_fig("Quality Component Scores by Rank", "components")
    add_fig("Avg FPS vs Overall Score", "fps_scatter")
    add_fig("Top Parameter Variation", "param_variation")
    add_fig("Numeric Parameter Heatmap", "param_heatmap_numeric")
    add_fig("Categorical Parameter Heatmap", "param_heatmap_categorical")

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
    figure.card {{ margin: 16px 0; }}
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

  <h2>Charts</h2>
  {''.join(chart_html_parts)}

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

    _ensure_dir(os.path.dirname(path))
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

    # 準備圖表輸出目錄
    img_dir = os.path.join(out_dir, f"filter_logs_figures_{timestamp}")
    _ensure_dir(img_dir)

    charts = {}

    def _safe_savefig(fig, path):
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _is_number(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
        try:
            fv = float(v)
            return not (math.isnan(fv) or math.isinf(fv))
        except Exception:
            return False

    # 構建 DataFrame 以便繪圖
    def _to_df(top):
        rows = []
        for idx, r in enumerate(top, 1):
            qs = r.get("quality_scores", {})
            km = r.get("key_metrics", {})
            row = {
                "rank": idx,
                "run_id": r.get("run_id"),
                "file": r.get("file_path"),
                "overall_score": qs.get("overall_score"),
                "continuity_score": qs.get("continuity_score"),
                "fragmentation_score": qs.get("fragmentation_score"),
                "efficiency_score": qs.get("efficiency_score"),
                "stability_score": qs.get("stability_score"),
                "avg_fps": km.get("avg_fps"),
                "continuity_ratio": km.get("continuity_ratio"),
                "fragmentation_ratio": km.get("fragmentation_ratio"),
                "id_switches": km.get("id_switches"),
                "total_tracks": km.get("total_tracks"),
                "avg_track_length": km.get("avg_track_length"),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    df_top = _to_df(top_list)
    # 圖 1：Top-K overall score bar
    if not df_top.empty and "overall_score" in df_top:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(
            data=df_top,
            x="rank",
            y="overall_score",
            palette="Blues_d",
            ax=ax,
        )
        ax.set_title("Top-K Overall Score")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Overall Score")
        p1 = os.path.join(img_dir, "topk_overall_score.png")
        _safe_savefig(fig, p1)
        charts["topk_overall"] = os.path.basename(p1)

    # 圖 2：四個品質分數的群組長條
    comp_cols = [
        "continuity_score",
        "fragmentation_score",
        "efficiency_score",
        "stability_score",
    ]
    if not df_top.empty and all(c in df_top.columns for c in comp_cols):
        dfm = df_top.melt(
            id_vars=["rank", "run_id"], value_vars=comp_cols,
            var_name="component", value_name="score"
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=dfm, x="rank", y="score", hue="component", ax=ax)
        ax.set_title("Quality Component Scores by Rank")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Score")
        ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")
        p2 = os.path.join(img_dir, "topk_components.png")
        _safe_savefig(fig, p2)
        charts["components"] = os.path.basename(p2)

    # 圖 3：avg_fps vs overall_score 散點
    if not df_top.empty and _is_number(df_top.get("avg_fps", [None]).iloc[0] if len(df_top) else None):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df_top["avg_fps"], df_top["overall_score"], c=df_top["rank"], cmap="viridis")
        for i, row in df_top.iterrows():
            ax.annotate(int(row["rank"]), (row["avg_fps"], row["overall_score"]), textcoords="offset points", xytext=(4,4), fontsize=8)
        ax.set_title("Avg FPS vs Overall Score")
        ax.set_xlabel("Avg FPS")
        ax.set_ylabel("Overall Score")
        p3 = os.path.join(img_dir, "fps_vs_overall.png")
        _safe_savefig(fig, p3)
        charts["fps_scatter"] = os.path.basename(p3)

    # 參數圖表
    def _collect_params(top):
        run_ids, params_list = [], []
        for r in top:
            run_ids.append(r.get("run_id"))
            params_list.append(r.get("parameters", {}) or {})
        return run_ids, params_list

    run_ids, params_list = _collect_params(top_list)
    # 合併所有鍵
    all_keys = []
    for p in params_list:
        for k in p.keys():
            if k not in all_keys:
                all_keys.append(k)

    # 統計唯一值數量，區分數值/非數值
    uniq_counts = []
    numeric_keys, categorical_keys = [], []
    for k in all_keys:
        vals = [p.get(k) for p in params_list]
        # 判斷是否數字（允許字串可轉浮點）
        if any(v is not None for v in vals) and all(_is_number(v) for v in vals if v is not None):
            numeric_keys.append(k)
        else:
            categorical_keys.append(k)
        uniq = len({json.dumps(v, sort_keys=True) for v in vals})
        uniq_counts.append((k, uniq))

    # 參數唯一值長條圖（顯示最有變化的前20個）
    if uniq_counts:
        uniq_counts.sort(key=lambda x: x[1], reverse=True)
        top_param_var = uniq_counts[: min(20, len(uniq_counts))]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar([k for k, _ in top_param_var], [c for _, c in top_param_var], color="#3498db")
        ax.set_title("Top Parameter Variation (unique values across Top-K)")
        ax.set_ylabel("Unique value count")
        ax.set_xticklabels([k for k, _ in top_param_var], rotation=45, ha="right")
        p4 = os.path.join(img_dir, "param_variation_bar.png")
        _safe_savefig(fig, p4)
        charts["param_variation"] = os.path.basename(p4)

    # 數值型參數熱圖（標準化 0-1）
    if numeric_keys:
        data = []
        for params in params_list:
            row = []
            for k in numeric_keys:
                v = params.get(k)
                row.append(float(v) if _is_number(v) else np.nan)
            data.append(row)
        arr = np.array(data, dtype=float)
        # min-max 標準化各列
        with np.errstate(invalid='ignore'):
            mins = np.nanmin(arr, axis=0)
            maxs = np.nanmax(arr, axis=0)
            denom = np.where(maxs - mins == 0, 1, (maxs - mins))
            norm = (arr - mins) / denom
        fig, ax = plt.subplots(figsize=(max(8, len(numeric_keys) * 0.5), max(3, len(top_list) * 0.4)))
        sns.heatmap(norm, cmap="mako", cbar=True, ax=ax, vmin=0, vmax=1)
        ax.set_title("Numeric Parameter Heatmap (min-max normalized)")
        ax.set_xticks(np.arange(len(numeric_keys)) + 0.5)
        ax.set_xticklabels(numeric_keys, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(run_ids)) + 0.5)
        ax.set_yticklabels([f"#{i+1}" for i in range(len(run_ids))])
        p5 = os.path.join(img_dir, "param_numeric_heatmap.png")
        _safe_savefig(fig, p5)
        charts["param_heatmap_numeric"] = os.path.basename(p5)

    # 類別型參數熱圖（類別編碼 -> 0..n，再 / (n-1) 映射到 0-1）
    if categorical_keys:
        # 建立每欄的類別映射
        cat_maps = {}
        for k in categorical_keys:
            vals = [json.dumps(p.get(k, None), sort_keys=True) for p in params_list]
            uniq = sorted(list(set(vals)))
            cat_maps[k] = {u: i for i, u in enumerate(uniq)}

        data = []
        for params in params_list:
            row = []
            for k in categorical_keys:
                v = json.dumps(params.get(k, None), sort_keys=True)
                row.append(float(cat_maps[k].get(v, np.nan)))
            data.append(row)
        arr = np.array(data, dtype=float)
        # normalize per column to 0..1 if more than one category
        with np.errstate(invalid='ignore'):
            maxs = np.nanmax(arr, axis=0)
            denom = np.where(maxs == 0, 1, maxs)
            norm = arr / denom
        fig, ax = plt.subplots(figsize=(max(8, len(categorical_keys) * 0.5), max(3, len(top_list) * 0.4)))
        sns.heatmap(norm, cmap="crest", cbar=True, ax=ax, vmin=0, vmax=1)
        ax.set_title("Categorical Parameter Heatmap (encoded)")
        ax.set_xticks(np.arange(len(categorical_keys)) + 0.5)
        ax.set_xticklabels(categorical_keys, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(run_ids)) + 0.5)
        ax.set_yticklabels([f"#{i+1}" for i in range(len(run_ids))])
        p6 = os.path.join(img_dir, "param_categorical_heatmap.png")
        _safe_savefig(fig, p6)
        charts["param_heatmap_categorical"] = os.path.basename(p6)

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
    # 將圖表嵌入 HTML
    # 以相對路徑引用圖檔，放在同一 out_dir 下的 img_dir
    charts_rel = {k: os.path.join(os.path.basename(img_dir), v) for k, v in charts.items()}
    _write_html_with_charts(
        html_path,
        timestamp,
        os.path.abspath(input_dir),
        summary.get("total_results", 0),
        json_out.get("best", {}),
        top_list,
        charts_rel,
    )

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
