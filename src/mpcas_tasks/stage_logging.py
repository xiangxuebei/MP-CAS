#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User-facing stage status logging for analysis pipelines."""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PIPELINE_VERBOSE = os.environ.get("PIPELINE_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}
RUNTIME_HISTORY_VERSION = 1

STAGE_ALIASES = {
    "gaze": {"gaze feature extraction", "\u773c\u52a8\u7279\u5f81\u63d0\u53d6"},
    "audio": {"audio feature extraction", "\u8bed\u97f3\u7279\u5f81\u63d0\u53d6"},
    "fusion": {"feature fusion", "\u6574\u7406\u7279\u5f81\u7ed3\u679c"},
    "prep": {"model preparation", "\u51c6\u5907\u5206\u6790\u73af\u5883"},
    "inference": {
        "model inference",
        "\u751f\u6210\u521d\u7b5b\u7ed3\u679c",
        "\u751f\u6210\u8fdb\u9636\u7ed3\u679c",
    },
}

HIDDEN_STAGE_NAMES = {
    "特征提取总流程",
    "分类模型训练总流程",
    "分类模型推理总流程",
    "端到端训练总流程",
    "音频依赖检查",
    "音频WAV抽取",
    "音频ASR与完整特征提取",
    "音频特征表加载",
    "数据加载",
    "模态特征表读取",
    "Coginter 提取交互特征",
    "多模态特征融合",
    "训练数据处理",
    "推理数据加载与对齐",
    "推理结果保存",
}

SUBJECT_FACING_STAGE_NAMES = {
    "Coggaze 提取眼动特征": "眼动特征提取",
    "Cogaudio 提取音频特征": "语音特征提取",
    "整理特征结果": "整理特征结果",
    "整理测评结果": "整理特征结果",
    "训练数据加载": "读取训练数据",
    "XGBoost模型训练": "训练分析模型",
    "训练结果保存": "保存模型结果",
    "推理模型加载": "准备分析环境",
    "模型推理（XGBoost）": "生成初筛结果",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except Exception:
        value = default
    if value <= minimum:
        return default
    return value


def env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except Exception:
        value = default
    if value <= minimum:
        return default
    return value


def atomic_write_json(path: Path, payload: Dict[str, Any], encoding: str = "utf-8") -> None:
    ensure_dir(path.parent)
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    last_error: Optional[PermissionError] = None
    for attempt in range(6):
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.{attempt}.tmp")
        try:
            tmp_path.write_text(raw, encoding=encoding)
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error


def read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def display_stage_name(name: str) -> Optional[str]:
    if name in HIDDEN_STAGE_NAMES:
        return None
    return SUBJECT_FACING_STAGE_NAMES.get(name, name)


def normalize_event_message(message: str) -> Optional[str]:
    text = str(message).strip()
    if not text:
        return None
    if text.startswith("[CONFIG]") or text.startswith("[RUN]") or text.startswith("[SKIP]"):
        return None
    if text.startswith("[PIPELINE") or text.startswith("[INTERNAL STAGE"):
        return None
    if text.startswith("[DATA]") or text.startswith("[FEATURES]") or text.startswith("[FUSION]"):
        return None
    if text.startswith("[TRAIN DATA]") or text.startswith("[TRAIN SAVE]"):
        return None
    if text.startswith("[PREDICT MODEL]") or text.startswith("[PREDICT DATA]") or text.startswith("[PREDICT SAVE]"):
        return None
    if text.startswith("[AUDIO WAV]") or text.startswith("[AUDIO]"):
        return None
    if re.search(r"[A-Za-z]:\\|/", text):
        return None
    if re.search(r"[\u4e00-\u9fff]", text):
        return text
    return None


def user_facing_error(error: str) -> str:
    text = str(error)
    if "missing trained model files" in text:
        return "未找到训练好的分析模型"
    if "base_score must be in (0,1)" in text or "logistic loss" in text:
        return "训练数据不足或标签类别不完整，无法训练分析模型"
    if "feature CSV must contain subject" in text:
        return "测评特征文件缺少受试者信息"
    if "feature CSV must include label" in text:
        return "训练数据缺少分组标签"
    return "该步骤处理失败"


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {sec:05.2f}s"
    if minutes:
        return f"{minutes:d}m {sec:05.2f}s"
    return f"{sec:.2f}s"


def parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.astimezone()
    return parsed


def approximate_minutes(seconds: float) -> int:
    seconds = max(0.0, float(seconds))
    return int(round(seconds / 60.0))


def whole_seconds(seconds: float) -> int:
    return max(0, int(round(max(0.0, float(seconds)))))


def format_duration_zh(seconds: float) -> str:
    total = whole_seconds(seconds)
    minutes, sec = divmod(total, 60)
    if minutes <= 0:
        return f"{sec} 秒"
    return f"{minutes} 分钟 {sec} 秒" if sec else f"{minutes} 分钟"


def seconds_between(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def stage_estimate_seconds(data: Dict[str, Any], name: Any) -> float:
    estimates = data.get("stage_estimates")
    if not isinstance(estimates, dict):
        return 0.0
    try:
        return max(0.0, float(estimates.get(str(name)) or 0.0))
    except Exception:
        return 0.0


def estimated_remaining_seconds_for(data: Dict[str, Any], now: datetime) -> Optional[float]:
    steps = data.get("steps")
    estimates = data.get("stage_estimates")
    if not isinstance(steps, list) or not isinstance(estimates, dict) or not estimates:
        return None

    remaining = 0.0
    used_estimate = False
    for step in steps:
        if not isinstance(step, dict):
            continue
        status = str(step.get("status") or "pending").lower()
        if status in {"done", "failed", "cancelled"}:
            continue
        estimate = stage_estimate_seconds(data, step.get("name"))
        if estimate <= 0:
            continue
        used_estimate = True
        if status == "running":
            started = parse_iso_datetime(step.get("started_at"))
            elapsed = seconds_between(started, now) or 0.0
            if elapsed <= estimate:
                remaining += estimate - elapsed
            else:
                remaining += max(30.0, min(estimate, elapsed * 0.25))
        else:
            remaining += estimate
    return remaining if used_estimate else None


def progress_hint_for(data: Dict[str, Any]) -> Optional[str]:
    return progress_metrics_for(data).get("progress_hint")


def progress_metrics_for(data: Dict[str, Any]) -> Dict[str, Any]:
    status = str(data.get("status") or "").lower()
    started = parse_iso_datetime(data.get("submitted_at")) or parse_iso_datetime(data.get("started_at"))
    if started is None:
        return {
            "progress_hint": None,
            "elapsed_seconds": None,
            "elapsed_minutes": None,
            "remaining_seconds": None,
            "remaining_minutes": None,
        }
    updated = parse_iso_datetime(data.get("updated_at")) or datetime.now().astimezone()
    try:
        estimated_minutes = int(data.get("estimated_minutes") or 0)
    except Exception:
        estimated_minutes = 0
    if estimated_minutes <= 0:
        estimated_minutes = 10 if str(data.get("assessment_phase") or "").lower() == "advanced" else 5

    if status == "running":
        now = datetime.now().astimezone()
        elapsed_seconds = max(0.0, (now - started).total_seconds())
        remaining_seconds = estimated_remaining_seconds_for(data, now)
        if remaining_seconds is None:
            estimated_seconds = float(data.get("estimated_seconds") or estimated_minutes * 60)
            remaining_seconds = max(0.0, estimated_seconds - elapsed_seconds)
        return {
            "progress_hint": f"已运行 {format_duration_zh(elapsed_seconds)}，预计还需 {format_duration_zh(remaining_seconds)}",
            "elapsed_seconds": whole_seconds(elapsed_seconds),
            "elapsed_minutes": approximate_minutes(elapsed_seconds),
            "remaining_seconds": whole_seconds(remaining_seconds),
            "remaining_minutes": approximate_minutes(remaining_seconds),
        }
    if status == "done":
        elapsed_seconds = max(0.0, (updated - started).total_seconds())
        return {
            "progress_hint": f"共耗时 {format_duration_zh(elapsed_seconds)}",
            "elapsed_seconds": whole_seconds(elapsed_seconds),
            "elapsed_minutes": approximate_minutes(elapsed_seconds),
            "remaining_seconds": 0,
            "remaining_minutes": 0,
        }
    if status in {"failed", "cancelled"}:
        elapsed_seconds = max(0.0, (updated - started).total_seconds())
        return {
            "progress_hint": f"已运行 {format_duration_zh(elapsed_seconds)}，分析未完成",
            "elapsed_seconds": whole_seconds(elapsed_seconds),
            "elapsed_minutes": approximate_minutes(elapsed_seconds),
            "remaining_seconds": None,
            "remaining_minutes": None,
        }
    return {
        "progress_hint": None,
        "elapsed_seconds": None,
        "elapsed_minutes": None,
        "remaining_seconds": None,
        "remaining_minutes": None,
    }


def runtime_history_path_from_status(log_path: Path) -> Optional[Path]:
    try:
        phase_dir = log_path.parent
        subject_dir = phase_dir.parent
        result_root = subject_dir.parent
    except Exception:
        return None
    if not phase_dir.name or not subject_dir.name or result_root == subject_dir:
        return None
    return result_root / "_runtime_history.json"


def stage_key_for_name(name: Any) -> Optional[str]:
    text = str(name or "")
    for key, aliases in STAGE_ALIASES.items():
        if text in aliases:
            return key
    return None


def step_duration_seconds(step: Dict[str, Any]) -> Optional[float]:
    started = parse_iso_datetime(step.get("started_at"))
    finished = parse_iso_datetime(step.get("finished_at"))
    return seconds_between(started, finished)


def completed_stage_durations(data: Dict[str, Any]) -> Dict[str, float]:
    durations: Dict[str, float] = {}
    for step in data.get("steps") or []:
        if not isinstance(step, dict) or str(step.get("status") or "").lower() != "done":
            continue
        stage_key = stage_key_for_name(step.get("name"))
        duration = step_duration_seconds(step)
        if stage_key and duration is not None:
            durations[stage_key] = durations.get(stage_key, 0.0) + duration
    return durations


def update_ema(old_value: Any, new_value: float, alpha: float) -> float:
    try:
        old = float(old_value)
    except Exception:
        old = 0.0
    if old <= 0:
        return new_value
    return old * (1.0 - alpha) + new_value * alpha


def update_stage_history(
    history: Dict[str, Any],
    stage: str,
    seconds: float,
    media_mb: Optional[float] = None,
) -> None:
    if seconds <= 0:
        return
    alpha = min(1.0, max(0.01, env_float("AINAO_RUNTIME_HISTORY_ALPHA", 0.35, minimum=0.0)))
    stages = history.setdefault("stages", {})
    item = stages.setdefault(stage, {})
    samples = int(item.get("samples") or 0) + 1
    max_samples = env_int("AINAO_RUNTIME_HISTORY_MAX_SAMPLES", 100, minimum=1)
    item["samples"] = min(samples, max_samples)
    item["seconds_ema"] = round(update_ema(item.get("seconds_ema"), seconds, alpha), 3)
    item["last_seconds"] = round(seconds, 3)
    if media_mb is not None and media_mb > 0 and seconds > 0:
        speed = media_mb / seconds
        item["mb_per_sec_ema"] = round(update_ema(item.get("mb_per_sec_ema"), speed, alpha), 3)
        item["last_mb_per_sec"] = round(speed, 3)
        item["last_media_mb"] = round(media_mb, 3)


def update_runtime_history_for_done_task(log_path: Path, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    input_profile = data.get("input_profile")
    if not isinstance(input_profile, dict):
        return None
    history_path = runtime_history_path_from_status(log_path)
    if history_path is None:
        return None

    history = read_json_file(history_path) or {"version": RUNTIME_HISTORY_VERSION, "stages": {}}
    if int(history.get("version") or 0) != RUNTIME_HISTORY_VERSION:
        history = {"version": RUNTIME_HISTORY_VERSION, "stages": {}}

    task_id = str(data.get("analysis_task_id") or "")
    recent_ids = [str(item) for item in history.get("recent_analysis_task_ids") or []]
    if task_id and task_id in recent_ids:
        return history

    durations = completed_stage_durations(data)
    if "gaze" in durations:
        update_stage_history(history, "gaze", durations["gaze"], float(input_profile.get("gaze_media_mb") or 0.0))
    if "audio" in durations:
        update_stage_history(history, "audio", durations["audio"], float(input_profile.get("audio_media_mb") or 0.0))
    for stage in ["fusion", "prep", "inference"]:
        if stage in durations:
            update_stage_history(history, stage, durations[stage])

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    history["version"] = RUNTIME_HISTORY_VERSION
    history["updated_at"] = now
    history["last_analysis_task_id"] = task_id or None
    if task_id:
        history["recent_analysis_task_ids"] = [task_id, *recent_ids][:100]
    atomic_write_json(history_path, history)
    data["runtime_history_updated_at"] = now
    data["runtime_history_path"] = str(history_path)
    return history


class StageLogger:
    def __init__(
        self,
        log_path: Path,
        analysis_task_id: str,
        subject_id: str,
        assessment_phase: str,
        timestamp: Optional[str] = None,
        planned_steps: Optional[List[str]] = None,
    ):
        self.log_path = log_path
        self.stage_stack: List[int] = []
        self.planned_step_names = set(planned_steps or [])
        self.assessment_phase = str(assessment_phase or "").strip().lower()
        ensure_dir(self.log_path.parent)
        now = self.now_iso()
        timestamp = timestamp or datetime.now().strftime("%Y%m%dT%H%M%S")
        existing_payload: Dict[str, Any] = {}
        if self.log_path.exists():
            try:
                loaded = json.loads(self.log_path.read_text(encoding="utf-8-sig"))
                if isinstance(loaded, dict):
                    existing_payload = loaded
            except Exception:
                existing_payload = {}
        existing_data = existing_payload.get("data") if isinstance(existing_payload.get("data"), dict) else {}
        steps = [
            {
                "step_index": idx,
                "name": name,
                "status": "pending",
                "started_at": None,
                "finished_at": None,
            }
            for idx, name in enumerate(planned_steps or [], start=1)
        ]
        events = existing_data.get("events") if isinstance(existing_data.get("events"), list) else []
        preserved_keys = [
            "submitted_at",
            "estimated_minutes",
            "estimated_seconds",
            "estimate_source",
            "estimate_sources",
            "stage_estimates",
            "input_profile",
            "runtime_profile",
            "runtime_history_summary",
            "cancelled_task_id",
            "result_path",
            "status_path",
            "analysis_log_path",
            "model_dir",
            "rawdata_selection",
            "pid",
        ]
        data = {key: existing_data[key] for key in preserved_keys if key in existing_data}
        data.update(
            {
                "analysis_task_id": analysis_task_id,
                "subject_id": subject_id,
                "assessment_phase": assessment_phase,
                "timestamp": timestamp,
                "status": "running",
                "started_at": existing_data.get("started_at") or now,
                "updated_at": now,
                "steps": steps,
                "current_step": None,
                "events": events,
            }
        )
        data.setdefault("estimated_minutes", 10 if self.assessment_phase == "advanced" else 5)
        self.payload: Dict[str, Any] = {
            "code": 0,
            "data": data,
        }
        self._save()

    @staticmethod
    def now_iso() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _sync_from_disk(self) -> None:
        if not self.log_path.exists():
            return
        try:
            loaded = json.loads(self.log_path.read_text(encoding="utf-8-sig"))
        except Exception:
            return
        if isinstance(loaded, dict) and isinstance(loaded.get("data"), dict):
            self.payload = loaded

    def _save(self) -> None:
        self._update_progress_hint()
        atomic_write_json(self.log_path, self.payload, encoding="utf-8-sig")

    def _data(self) -> Dict[str, Any]:
        data = self.payload.setdefault("data", {})
        data.setdefault("steps", [])
        data.setdefault("events", [])
        return data

    def _update_progress_hint(self) -> None:
        data = self._data()
        data.update(progress_metrics_for(data))
        if str(data.get("status") or "").lower() == "running":
            data["progress_updated_at"] = self.now_iso()

    def _touch(self) -> str:
        now = self.now_iso()
        self._data()["updated_at"] = now
        return now

    def _add_event(self, message: str) -> None:
        event_message = normalize_event_message(message)
        if not event_message:
            return
        self._data()["events"].append({"timestamp": self.now_iso(), "message": event_message})

    def _find_pending_step(self, name: str) -> Optional[Dict[str, Any]]:
        for step in self._data()["steps"]:
            if step.get("name") == name and step.get("status") == "pending":
                return step
        return None

    def _find_step(self, step_index: int) -> Optional[Dict[str, Any]]:
        for step in self._data()["steps"]:
            if int(step.get("step_index", -1)) == int(step_index):
                return step
        return None

    def _new_step(self, name: str) -> Dict[str, Any]:
        steps = self._data()["steps"]
        next_index = max([int(step.get("step_index", 0)) for step in steps] or [0]) + 1
        step = {
            "step_index": next_index,
            "name": name,
            "status": "pending",
            "started_at": None,
            "finished_at": None,
        }
        steps.append(step)
        return step

    def _set_current_step(self) -> None:
        steps = self._data()["steps"]
        current = None
        for step_index in reversed(self.stage_stack):
            step = self._find_step(step_index)
            if step and step.get("status") == "running":
                current = {"step_index": step["step_index"], "name": step["name"]}
                break
        if current is None:
            running = [step for step in steps if step.get("status") == "running"]
            if running:
                step = running[-1]
                current = {"step_index": step["step_index"], "name": step["name"]}
        self._data()["current_step"] = current

    def _display_stage_name(self, name: str) -> Optional[str]:
        display_name = display_stage_name(name)
        if display_name == "生成初筛结果" and self.assessment_phase == "advanced":
            return "生成进阶结果"
        return display_name

    def log(self, message: str) -> None:
        self._sync_from_disk()
        self._touch()
        self._add_event(message)
        self._save()

    def finish(self, status: str = "done", error: Optional[str] = None) -> None:
        self._sync_from_disk()
        data = self._data()
        data["status"] = status
        data["updated_at"] = self.now_iso()
        data["current_step"] = None
        if status == "failed":
            self.payload["code"] = 1
            if error:
                data["error"] = user_facing_error(error)
            self._add_event("分析流程失败")
        elif status == "done":
            self._add_event("分析流程完成")
        if status == "done":
            update_runtime_history_for_done_task(self.log_path, data)
        self._save()

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        display_name = self._display_stage_name(name)
        if display_name is None or (self.planned_step_names and display_name not in self.planned_step_names):
            if PIPELINE_VERBOSE:
                print(f"[STAGE START] {name}", flush=True)
            self.log(f"[INTERNAL STAGE START] {name}")
            try:
                yield
            except Exception as exc:
                elapsed = time.perf_counter() - start
                self.log(f"[INTERNAL STAGE FAILED] {name} elapsed={format_seconds(elapsed)} error={exc}")
                print(f"[STAGE FAILED] {name} | elapsed={format_seconds(elapsed)} | error={exc}", flush=True)
                raise
            else:
                elapsed = time.perf_counter() - start
                self.log(f"[INTERNAL STAGE END] {name} elapsed={format_seconds(elapsed)}")
                if PIPELINE_VERBOSE:
                    print(f"[STAGE END] {name} | elapsed={format_seconds(elapsed)}", flush=True)
                return

        self._sync_from_disk()
        step = self._find_pending_step(display_name) or self._new_step(display_name)
        now = self._touch()
        step["status"] = "running"
        step["started_at"] = step.get("started_at") or now
        step["finished_at"] = None
        step_index = int(step["step_index"])
        self.stage_stack.append(step_index)
        self._set_current_step()
        self._add_event(f"开始：{display_name}")
        self._save()
        print(f"[STAGE START] {name}", flush=True)
        try:
            yield
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._sync_from_disk()
            step = self._find_step(step_index)
            if step:
                step["status"] = "failed"
                step["finished_at"] = self._touch()
                step["error"] = user_facing_error(str(exc))
                self._add_event(f"失败：{display_name}")
            self.payload["code"] = 1
            self._data()["status"] = "failed"
            if step_index in self.stage_stack:
                self.stage_stack.remove(step_index)
            self._set_current_step()
            self._save()
            print(f"[STAGE FAILED] {name} | elapsed={format_seconds(elapsed)} | error={exc}", flush=True)
            raise
        else:
            elapsed = time.perf_counter() - start
            self._sync_from_disk()
            step = self._find_step(step_index)
            if step:
                step["status"] = "done"
                step["finished_at"] = self._touch()
                self._add_event(f"完成：{display_name}")
            if step_index in self.stage_stack:
                self.stage_stack.remove(step_index)
            self._set_current_step()
            self._save()
            print(f"[STAGE END] {name} | elapsed={format_seconds(elapsed)}", flush=True)


def optional_stage(logger: Optional[StageLogger], name: str):
    return logger.stage(name) if logger else nullcontext()
