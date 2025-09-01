"""
並行測試執行器
支持多進程和批量處理優化
"""

import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable
import logging
from dataclasses import dataclass


@dataclass
class TestTask:
    """測試任務"""
    task_id: str
    parameters: Dict[str, Any]
    config_path: str
    priority: int = 0


class ParallelTestExecutor:
    """並行測試執行器"""

    def __init__(self, max_workers: int = None, use_gpu_queue: bool = True):
        """
        初始化並行執行器

        Args:
            max_workers: 最大工作進程數
            use_gpu_queue: 是否使用 GPU 隊列管理
        """
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        self.use_gpu_queue = use_gpu_queue
        self.logger = logging.getLogger(__name__)

        # GPU 資源管理
        self.available_gpus = self._detect_gpus()
        self.gpu_queue = multiprocessing.Queue()

        if self.use_gpu_queue and self.available_gpus:
            for gpu_id in self.available_gpus:
                self.gpu_queue.put(gpu_id)

    def _detect_gpus(self) -> List[int]:
        """檢測可用的 GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
        except ImportError:
            pass
        return []

    def execute_tests_parallel(self, tasks: List[TestTask],
                             execute_func: Callable) -> List[Dict[str, Any]]:
        """
        並行執行測試任務

        Args:
            tasks: 測試任務列表
            execute_func: 執行函數

        Returns:
            測試結果列表
        """
        results = []

        # 按優先級排序
        tasks.sort(key=lambda x: x.priority, reverse=True)

        if len(tasks) == 1 or self.max_workers == 1:
            # 單線程執行
            for task in tasks:
                result = self._execute_single_task(task, execute_func)
                results.append(result)
        else:
            # 多進程執行
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self._execute_task_with_gpu, task, execute_func): task
                    for task in tasks
                }

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result(timeout=300)  # 5分鐘超時
                        results.append(result)
                        self.logger.info(f"任務 {task.task_id} 完成")
                    except Exception as e:
                        self.logger.error(f"任務 {task.task_id} 失敗: {e}")
                        results.append({
                            "task_id": task.task_id,
                            "error": str(e),
                            "status": "failed"
                        })

        return results

    def _execute_task_with_gpu(self, task: TestTask, execute_func: Callable) -> Dict[str, Any]:
        """帶 GPU 管理的任務執行"""
        gpu_id = None

        if self.use_gpu_queue and not self.gpu_queue.empty():
            try:
                gpu_id = self.gpu_queue.get(timeout=1)
            except:
                pass

        try:
            result = self._execute_single_task(task, execute_func, gpu_id)
            return result
        finally:
            # 歸還 GPU 資源
            if gpu_id is not None and self.use_gpu_queue:
                self.gpu_queue.put(gpu_id)

    def _execute_single_task(self, task: TestTask, execute_func: Callable,
                           gpu_id: int = None) -> Dict[str, Any]:
        """執行單個測試任務"""
        start_time = time.time()

        try:
            # 設定 GPU
            if gpu_id is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            result = execute_func(task.parameters, task.config_path, task.task_id)
            result['execution_time'] = time.time() - start_time
            result['gpu_id'] = gpu_id
            result['status'] = 'completed'

            return result

        except Exception as e:
            return {
                'task_id': task.task_id,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'gpu_id': gpu_id,
                'status': 'failed'
            }

    def optimize_task_batching(self, tasks: List[TestTask],
                              batch_size: int = None) -> List[List[TestTask]]:
        """
        優化任務批次處理

        Args:
            tasks: 任務列表
            batch_size: 批次大小

        Returns:
            批次化的任務列表
        """
        if batch_size is None:
            batch_size = max(1, len(tasks) // self.max_workers)

        batches = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batches.append(batch)

        return batches

    def monitor_resource_usage(self) -> Dict[str, Any]:
        """監控資源使用情況"""
        import psutil

        resource_info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
        }

        # GPU 資源監控
        if self.available_gpus:
            try:
                import torch
                resource_info['gpu_info'] = []
                for gpu_id in self.available_gpus:
                    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                    gpu_memory_used = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                    resource_info['gpu_info'].append({
                        'gpu_id': gpu_id,
                        'total_memory_gb': gpu_memory,
                        'used_memory_gb': gpu_memory_used,
                        'utilization_percent': (gpu_memory_used / gpu_memory) * 100
                    })
            except:
                pass

        return resource_info


class PerformanceProfiler:
    """效能分析器"""

    def __init__(self):
        self.profiles = {}

    def profile_function(self, func_name: str):
        """裝飾器：分析函數效能"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()

                result = func(*args, **kwargs)

                end_time = time.time()
                end_memory = self._get_memory_usage()

                self.profiles[func_name] = {
                    'execution_time': end_time - start_time,
                    'memory_delta_mb': end_memory - start_memory,
                    'calls': self.profiles.get(func_name, {}).get('calls', 0) + 1
                }

                return result
            return wrapper
        return decorator

    def _get_memory_usage(self) -> float:
        """獲取記憶體使用量（MB）"""
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)

    def get_performance_report(self) -> Dict[str, Any]:
        """獲取效能報告"""
        return {
            'function_profiles': self.profiles,
            'total_functions': len(self.profiles),
            'total_calls': sum(p.get('calls', 0) for p in self.profiles.values()),
            'total_time': sum(p.get('execution_time', 0) for p in self.profiles.values())
        }
