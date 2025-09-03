"""
測試執行器：呼叫 mot-infer 執行單次推理測試，並彙整結果。
"""

import subprocess
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import yaml


def generate_run_id() -> str:
    """產生唯一的 run_id，用於綁定影片與日誌等產物。"""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    short = uuid.uuid4().hex[:8]
    return f"{ts}_{short}"


class TestExecutor:
    """測試執行器"""

    def __init__(self, base_config: Dict[str, Any], repo_root: Optional[str] = None):
        """
        初始化測試執行器

        Args:
            base_config: 基礎配置
            repo_root: 倉庫根目錄（執行 mot-infer 的 cwd），預設為自動推測
        """
        self.base_config = base_config
        # 推測倉庫根目錄：從當前檔案回到專案根（src/../../）
        if repo_root is None:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            self.repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
        else:
            self.repo_root = os.path.abspath(repo_root)

    def execute_test(self, config_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        執行單個測試

        Args:
            config_path: 測試配置檔案路徑
            parameters: 測試參數

        Returns:
            測試結果字典
        """
        # 生成唯一的 run_id
        run_id = generate_run_id()

        # 準備命令參數
        cmd_args, env_vars = self._prepare_command_args(config_path, run_id)

        # 執行測試
        start_time = datetime.now()
        result = self._run_command(cmd_args, env_vars)
        end_time = datetime.now()

        # 收集結果
        test_result = {
            "run_id": run_id,
            "parameters": parameters,
            "execution_info": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "command": " ".join(cmd_args),
                "exit_code": result.returncode
            },
            "file_paths": self._get_output_paths(run_id)
        }

        # 如果執行成功，嘗試解析結果
        if result.returncode == 0:
            test_result.update(self._parse_execution_results(run_id))
        else:
            test_result["error"] = result.stderr.decode('utf-8', errors='ignore')

        return test_result

    def _prepare_command_args(self, config_path: str, run_id: str) -> List[str]:
        """
        準備命令參數

        Args:
            config_path: 配置檔案路徑
            run_id: 運行 ID

        Returns:
            命令參數列表
        """
        # 以 CLI 方式呼叫 mot-infer，盡量用參數避免倚賴環境變數
        weights = self.base_config.get("weights", "models/yolo12x.pt")
        source = self.base_config.get("video_source", "data/samples/camera-d8accaf0_10s.mp4")
        output = self.base_config.get("output_dir", "outputs/videos")
        track_config_name = self.base_config.get("track_config_name", "default")
        track_configs_path = self.base_config.get("track_configs_path", "configs/track_configs.json")

        args: List[str] = [
            "mot-infer",
            "--weights", weights,
            "--source", source,
            "--output", output,
            "--track-config", track_config_name,
            "--track-configs-path", track_configs_path,
            "--run-id", run_id,
        ]
        overrides = self._prepare_track_overrides(config_path)
        if overrides:
            args += ["--track-overrides", overrides]

        # 不必設定環境變數；保留空字典
        env_vars: Dict[str, str] = {}
        return args, env_vars

    def _prepare_track_overrides(self, config_path: str) -> str:
        """
        準備追蹤覆寫參數

        Args:
            config_path: 配置檔案路徑

        Returns:
            覆寫參數字串
        """
        # 只設置 tracker 參數指向我們的臨時配置檔案
        # YOLO 會自動讀取該檔案作為追蹤器配置
        return f"tracker='{config_path}'"

    def _run_command(self, cmd_args: List[str], env_vars: Dict[str, str]) -> subprocess.CompletedProcess:
        """
        執行命令

        Args:
            cmd_args: 命令參數
            env_vars: 環境變數

        Returns:
            執行結果
        """
        try:
            # 合併環境變數
            env = os.environ.copy()
            env.update(env_vars)

            result = subprocess.run(
                cmd_args,
                cwd=self.repo_root,
                capture_output=True,
                text=False,  # 保持為 bytes
                timeout=3600,  # 1小時超時
                env=env  # 使用更新的環境變數
            )
            return result
        except subprocess.TimeoutExpired:
            # 超時處理
            return subprocess.CompletedProcess(
                cmd_args, -1,
                stdout=b"", stderr=b"Test execution timed out"
            )

    def _get_output_paths(self, run_id: str) -> Dict[str, str]:
        """
        獲取輸出檔案路徑

        Args:
            run_id: 運行 ID

        Returns:
            檔案路徑字典
        """
        output = self.base_config.get("output_dir", "outputs/videos")
        source = self.base_config.get("video_source", "data/samples/camera-d8accaf0_60s.mp4")
        stem = os.path.splitext(os.path.basename(source))[0]
        return {
            "video_path": os.path.join(self.repo_root, output, f"{stem}__{run_id}.mp4"),
            "log_path": os.path.join(self.repo_root, "logs", f"inference_{run_id}.log"),
            "metadata_path": os.path.join(self.repo_root, output, f"{stem}__{run_id}.json"),
        }

    def _parse_execution_results(self, run_id: str) -> Dict[str, Any]:
        """
        解析執行結果

        Args:
            run_id: 運行 ID

        Returns:
            解析後的結果
        """
        results: Dict[str, Any] = {}
        paths = self._get_output_paths(run_id)
        metadata_path = paths.get("metadata_path")
        try:
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    results["metadata"] = metadata
            else:
                results["metadata_error"] = f"No JSON found: {metadata_path}"
        except Exception as e:
            results["metadata_error"] = str(e)
        return results
