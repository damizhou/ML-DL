"""
AppScanner Multi-Dataset Concurrent Training Script

并发训练多个 AppScanner 数据集，最多同时运行 3 个训练任务。

默认会调用 `train_with_dataset.py`，每个子进程只处理一个 `.pkl` 特征文件。
如果默认数据文件不存在，会在汇总中标记为 `missing`，而不是直接崩溃。

Usage:
    python train_multi_datasets.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class DatasetConfig:
    """Single dataset configuration."""

    name: str
    data_path: Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_with_dataset.py"
OUTPUT_ROOT = SCRIPT_DIR / "output_multi"
MODEL_OUTPUT_ROOT = SCRIPT_DIR / "output"
MAX_WORKERS = 1
WAIT_FOR_PID: Optional[int] = 3663076
WAIT_CHECK_INTERVAL_SECONDS = 30 * 60


DATASETS: List[DatasetConfig] = [
    DatasetConfig("vpn", SCRIPT_DIR / "data" / "vpn" / "vpn_appscanner.pkl"),
    # DatasetConfig("novpn", SCRIPT_DIR / "data" / "novpn" / "novpn_appscanner.pkl"),
    DatasetConfig("novpn_top1000", SCRIPT_DIR / "data" / "novpn_top1000" / "novpn_top1000_appscanner.pkl"),
    DatasetConfig("vpn_top1000", SCRIPT_DIR / "data" / "vpn_top1000" / "vpn_top1000_appscanner.pkl"),
    DatasetConfig("novpn_top500", SCRIPT_DIR / "data" / "novpn_top500" / "novpn_top500_appscanner.pkl"),
    DatasetConfig("vpn_top500", SCRIPT_DIR / "data" / "vpn_top500" / "vpn_top500_appscanner.pkl"),
    DatasetConfig("novpn_top100", SCRIPT_DIR / "data" / "novpn_top100" / "novpn_top100_appscanner.pkl"),
    DatasetConfig("vpn_top100", SCRIPT_DIR / "data" / "vpn_top100" / "vpn_top100_appscanner.pkl"),
    DatasetConfig("novpn_top50", SCRIPT_DIR / "data" / "novpn_top50" / "novpn_top50_appscanner.pkl"),
    DatasetConfig("vpn_top50", SCRIPT_DIR / "data" / "vpn_top50" / "vpn_top50_appscanner.pkl"),
    DatasetConfig("novpn_top10", SCRIPT_DIR / "data" / "novpn_top10" / "novpn_top10_appscanner.pkl"),
    DatasetConfig("vpn_top10", SCRIPT_DIR / "data" / "vpn_top10" / "vpn_top10_appscanner.pkl"),
]


SPLIT_METRIC_PATTERNS = {
    "train": {
        "accuracy": (r"Train Accuracy:\s*([0-9]*\.?[0-9]+)",),
        "macro_f1": (r"Train Macro-F1:\s*([0-9]*\.?[0-9]+)",),
    },
    "val": {
        "accuracy": (r"Val Accuracy:\s*([0-9]*\.?[0-9]+)",),
        "macro_f1": (r"Val Macro-F1:\s*([0-9]*\.?[0-9]+)",),
    },
    "test": {
        "accuracy": (
            r"Test Accuracy:\s*([0-9]*\.?[0-9]+)",
            r"Overall Accuracy:\s*([0-9]*\.?[0-9]+)",
            r"Accuracy:\s*([0-9]*\.?[0-9]+)",
        ),
        "macro_f1": (
            r"Test Macro-F1:\s*([0-9]*\.?[0-9]+)",
            r"Macro-F1:\s*([0-9]*\.?[0-9]+)",
        ),
    },
}

WEIGHTED_F1_PATTERN = re.compile(r"\b(?:Train|Val|Test)?\s*F1 (?:Score )?\(weighted\):")


def _format_elapsed(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _extract_metric(text: str, patterns: Iterable[str]) -> Optional[float]:
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    return None


def _make_base_result(config: DatasetConfig) -> Dict[str, object]:
    subprocess_log = OUTPUT_ROOT / config.name / f"{config.name}_subprocess_output.log"
    return {
        "name": config.name,
        "data_path": str(config.data_path),
        "model_output_dir": str(MODEL_OUTPUT_ROOT / config.name),
        "status": "failed",
        "train_accuracy": None,
        "train_macro_f1": None,
        "val_accuracy": None,
        "val_macro_f1": None,
        "test_accuracy": None,
        "test_macro_f1": None,
        "accuracy": None,
        "macro_f1": None,
        "elapsed": "00:00:00",
        "error": None,
        "returncode": None,
        "subprocess_log": str(subprocess_log),
    }


def train_single_dataset(config: DatasetConfig) -> Dict[str, object]:
    """Train a single dataset by calling train_with_dataset.py."""
    start_time = datetime.now()
    output_dir = OUTPUT_ROOT / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data_path",
        str(config.data_path),
    ]

    result = _make_base_result(config)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    try:
        process = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        result["returncode"] = process.returncode
        combined_output = process.stdout
        if process.stderr:
            combined_output = f"{process.stdout}\n{process.stderr}"

        subprocess_log = output_dir / f"{config.name}_subprocess_output.log"
        subprocess_log.write_text(
            "\n".join(
                [
                    f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Command: {' '.join(cmd)}",
                    f"Dataset: {config.name}",
                    f"Data path: {config.data_path}",
                    f"Model output dir: {MODEL_OUTPUT_ROOT / config.name}",
                    "",
                    "STDOUT:",
                    process.stdout,
                    "",
                    "STDERR:",
                    process.stderr,
                ]
            ),
            encoding="utf-8",
        )

        if process.returncode == 0:
            result["status"] = "success"
            for split_name, split_patterns in SPLIT_METRIC_PATTERNS.items():
                result[f"{split_name}_accuracy"] = _extract_metric(
                    combined_output,
                    split_patterns["accuracy"],
                )
                result[f"{split_name}_macro_f1"] = _extract_metric(
                    combined_output,
                    split_patterns["macro_f1"],
                )
            result["accuracy"] = result["test_accuracy"]
            result["macro_f1"] = result["test_macro_f1"]
            if WEIGHTED_F1_PATTERN.search(combined_output) and (
                result["train_macro_f1"] is None
                or result["val_macro_f1"] is None
                or result["test_macro_f1"] is None
            ):
                result["status"] = "metric_error"
                result["error"] = (
                    "Weighted-F1 was reported but Macro-F1 was missing. "
                    "Do not use this run for Table 5-2; update engine.py/train_with_dataset.py "
                    "and rerun."
                )
            elif result["test_accuracy"] is None and result["test_macro_f1"] is None:
                result["error"] = "Process completed, but metrics could not be parsed from output."
        else:
            result["error"] = process.stderr.strip() or process.stdout.strip() or (
                f"Process exited with code {process.returncode}"
            )
            (output_dir / f"{config.name}_error.log").write_text(
                str(result["error"]),
                encoding="utf-8",
            )

    except Exception as exc:  # pragma: no cover - defensive wrapper
        result["error"] = str(exc)
        (output_dir / f"{config.name}_error.log").write_text(
            f"Exception: {exc}",
            encoding="utf-8",
        )

    elapsed = (datetime.now() - start_time).total_seconds()
    result["elapsed"] = _format_elapsed(elapsed)
    return result


def _missing_dataset_result(config: DatasetConfig) -> Dict[str, object]:
    result = _make_base_result(config)
    result["status"] = "missing"
    result["error"] = "Dataset file not found."
    return result


def _ordered_results(results_by_name: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    return [results_by_name[dataset.name] for dataset in DATASETS]


def _print_live_result(result: Dict[str, object]) -> None:
    status = str(result["status"]).upper()
    print(
        f"[{status:<7}] {result['name']:20s} | "
        f"Train: {_format_metric(result['train_accuracy']):>7}/"
        f"{_format_metric(result['train_macro_f1']):>7} | "
        f"Val: {_format_metric(result['val_accuracy']):>7}/"
        f"{_format_metric(result['val_macro_f1']):>7} | "
        f"Test: {_format_metric(result['test_accuracy']):>7}/"
        f"{_format_metric(result['test_macro_f1']):>7} | "
        f"Time: {result['elapsed']}"
    )


def _save_summary(
    results: List[Dict[str, object]],
    start_time: datetime,
    end_time: datetime,
) -> None:
    elapsed = _format_elapsed((end_time - start_time).total_seconds())

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_txt = OUTPUT_ROOT / "summary.txt"
    summary_json = OUTPUT_ROOT / "summary.json"

    lines = [
        "AppScanner Multi-Dataset Training Summary",
        "=" * 80,
        f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total: {elapsed}",
        "",
        (
            f"{'Dataset':<20} {'Status':<10} "
            f"{'Train Acc':>10} {'Train F1':>10} "
            f"{'Val Acc':>10} {'Val F1':>10} "
            f"{'Test Acc':>10} {'Test F1':>10} {'Time':>12}"
        ),
        "-" * 120,
    ]
    for result in results:
        lines.append(
            f"{result['name']:<20} "
            f"{str(result['status']):<10} "
            f"{_format_metric(result['train_accuracy']):>10} "
            f"{_format_metric(result['train_macro_f1']):>10} "
            f"{_format_metric(result['val_accuracy']):>10} "
            f"{_format_metric(result['val_macro_f1']):>10} "
            f"{_format_metric(result['test_accuracy']):>10} "
            f"{_format_metric(result['test_macro_f1']):>10} "
            f"{str(result['elapsed']):>12}"
        )

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_json.write_text(
        json.dumps(
            {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_elapsed": elapsed,
                "max_workers": MAX_WORKERS,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _is_process_running(pid: int) -> bool:
    """Return True when the target process is still alive."""
    if pid <= 0:
        return False

    try:
        import psutil  # type: ignore

        return psutil.pid_exists(pid)
    except ImportError:
        pass

    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            return False
        return str(pid) in result.stdout

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_process_exit(pid: int, check_interval_seconds: int = WAIT_CHECK_INTERVAL_SECONDS) -> None:
    """Block until the target process exits, checking at a fixed interval."""
    print(f"Waiting for PID {pid} to exit before starting AppScanner jobs.")
    print(f"Check interval: {check_interval_seconds} seconds")

    while _is_process_running(pid):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] PID {pid} is still running. Rechecking in {check_interval_seconds} seconds.")
        time.sleep(check_interval_seconds)

    print(f"PID {pid} has exited. Starting AppScanner jobs.")


def main() -> None:
    start_time = datetime.now()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AppScanner Multi-Dataset Concurrent Training")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configured datasets: {len(DATASETS)}")
    print(f"Max concurrent workers: {MAX_WORKERS}")
    print(f"Wrapper output directory: {OUTPUT_ROOT}")
    print(f"Train script: {TRAIN_SCRIPT}")
    print()
    print("Datasets:")

    ready_datasets: List[DatasetConfig] = []
    results_by_name: Dict[str, Dict[str, object]] = {}

    for dataset in DATASETS:
        exists = dataset.data_path.exists()
        status = "ready" if exists else "missing"
        print(f"  - {dataset.name:20s} [{status}] {dataset.data_path}")
        if exists:
            ready_datasets.append(dataset)
        else:
            results_by_name[dataset.name] = _missing_dataset_result(dataset)

    print()

    if ready_datasets:
        worker_count = min(MAX_WORKERS, len(ready_datasets))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_dataset = {
                executor.submit(train_single_dataset, dataset): dataset for dataset in ready_datasets
            }

            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive wrapper
                    result = _make_base_result(dataset)
                    result["status"] = "error"
                    result["error"] = str(exc)
                results_by_name[dataset.name] = result
                _print_live_result(result)
    else:
        print("No available dataset files were found. Nothing will be launched.")

    ordered_results = _ordered_results(results_by_name)
    end_time = datetime.now()
    total_elapsed = _format_elapsed((end_time - start_time).total_seconds())

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(
        f"{'Dataset':<20} {'Status':<10} "
        f"{'Train Acc':>10} {'Train F1':>10} "
        f"{'Val Acc':>10} {'Val F1':>10} "
        f"{'Test Acc':>10} {'Test F1':>10} {'Time':>12}"
    )
    print("-" * 120)
    for result in ordered_results:
        print(
            f"{result['name']:<20} "
            f"{str(result['status']):<10} "
            f"{_format_metric(result['train_accuracy']):>10} "
            f"{_format_metric(result['train_macro_f1']):>10} "
            f"{_format_metric(result['val_accuracy']):>10} "
            f"{_format_metric(result['val_macro_f1']):>10} "
            f"{_format_metric(result['test_accuracy']):>10} "
            f"{_format_metric(result['test_macro_f1']):>10} "
            f"{str(result['elapsed']):>12}"
        )
    print("-" * 120)
    print(f"Total time: {total_elapsed}")
    success_count = sum(1 for result in ordered_results if result["status"] == "success")
    print(f"Successful: {success_count}/{len(ordered_results)}")

    _save_summary(ordered_results, start_time, end_time)
    print(f"Summary saved to: {OUTPUT_ROOT / 'summary.txt'}")
    print(f"JSON summary saved to: {OUTPUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    if WAIT_FOR_PID is not None:
        wait_for_process_exit(WAIT_FOR_PID, check_interval_seconds=WAIT_CHECK_INTERVAL_SECONDS)
    main()
