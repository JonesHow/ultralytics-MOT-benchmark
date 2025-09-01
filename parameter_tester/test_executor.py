"""
測import subprocess
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import yaml責調用 ultralytics_inference_video.py 執行測試
"""

import subprocess
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import yaml
import json


class TestExecutor:
    """測試執行器"""

    def __init__(self, script_path: str, base_config: Dict[str, Any]):
        """
        初始化測試執行器

        Args:
            script_path: ultralytics_inference_video.py 的路徑
            base_config: 基礎配置
        """
        self.script_path = script_path
        self.base_config = base_config

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
        run_id = str(uuid.uuid4())[:8]

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
        # 由於 ultralytics_inference_video.py 使用環境變數，我們需要設置環境變數
        env_vars = {
            "TRACK_CONFIG": self.base_config.get("track_config_name", "default"),
            "TRACK_CONFIGS_PATH": self.base_config.get("track_configs_path", "track_configs.json"),
            "TRACK_OVERRIDES": self._prepare_track_overrides(config_path)
        }

        # 命令只包含 python 和腳本路徑
        args = ["python", self.script_path]

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
                cwd=os.path.dirname(self.script_path),
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
        base_dir = os.path.dirname(self.script_path)

        return {
            "video_path": os.path.join(base_dir, "output_videos", f"camera-d8accaf0_60s__{run_id}.mp4"),
            "log_path": os.path.join(base_dir, "logs", f"inference_{run_id}.log"),
            "metadata_path": os.path.join(base_dir, "output_videos", f"camera-d8accaf0_60s__{run_id}.json")
        }

    def _parse_execution_results(self, run_id: str) -> Dict[str, Any]:
        """
        解析執行結果

        Args:
            run_id: 運行 ID

        Returns:
            解析後的結果
        """
        results = {}

        # 嘗試讀取最新的 metadata JSON 檔案
        # 由於我們不知道確切的 run_id，我們查找最新的檔案
        metadata_dir = os.path.join(os.path.dirname(self.script_path), "output_videos")
        if os.path.exists(metadata_dir):
            json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json') and 'camera-d8accaf0_60s' in f]
            if json_files:
                # 按修改時間排序，獲取最新的
                json_files.sort(key=lambda x: os.path.getmtime(os.path.join(metadata_dir, x)), reverse=True)
                latest_json = json_files[0]
                metadata_path = os.path.join(metadata_dir, latest_json)

                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        results["metadata"] = metadata
                except Exception as e:
                    results["metadata_error"] = str(e)

        return results
