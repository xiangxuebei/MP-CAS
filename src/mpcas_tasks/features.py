#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multimodal feature table construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    from mpcas_tasks.records import PhaseRecord, discover_phase_records, parse_private_subject
    from mpcas_tasks.stage_logging import StageLogger, optional_stage
except ImportError:  # pragma: no cover - supports direct script execution
    from records import PhaseRecord, discover_phase_records, parse_private_subject
    from stage_logging import StageLogger, optional_stage


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_name(value: Any) -> str:
    text = str(value).strip()
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "field"


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def flatten_json_numeric(obj: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_clean = clean_name(key)
            new_prefix = f"{prefix}_{key_clean}" if prefix else key_clean
            out.update(flatten_json_numeric(value, new_prefix))
        return out
    if isinstance(obj, list):
        numeric_values: List[float] = []
        for idx, value in enumerate(obj):
            if isinstance(value, (dict, list)):
                out.update(flatten_json_numeric(value, f"{prefix}_{idx}"))
                continue
            maybe = to_float(value)
            if maybe is not None:
                numeric_values.append(maybe)
        if numeric_values:
            arr = np.asarray(numeric_values, dtype=float)
            out[f"{prefix}_count"] = float(len(arr))
            out[f"{prefix}_mean"] = float(np.nanmean(arr))
            out[f"{prefix}_std"] = float(np.nanstd(arr))
            out[f"{prefix}_min"] = float(np.nanmin(arr))
            out[f"{prefix}_max"] = float(np.nanmax(arr))
        return out

    value = to_float(obj)
    if value is not None and prefix:
        out[prefix] = value
    return out


def make_row_key(task: str, phase: str) -> str:
    return f"{clean_name(task)}__{clean_name(phase)}"


def is_feature_column(df: pd.DataFrame, col: str) -> bool:
    if col in {
        "subject",
        "label",
        "folder_subject",
        "subject_code",
        "device_id",
        "version",
        "task",
        "phase",
        "calibration_key",
        "record_key",
        "video_path",
        "source_root",
        "source_video",
        "status",
        "error",
        "orientation_used",
        "subject_raw",
        "subject_bucket",
        "subject_dir_name",
        "device",
    }:
        return False
    return True


def wide_from_long(
    df: pd.DataFrame,
    modality: str,
    subject_col: str,
    task_col: str,
    phase_col: str,
    extra_drop_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["subject"])
    work = df.copy()
    work[subject_col] = work[subject_col].astype(str)
    extra_drop = set(extra_drop_cols or [])
    rows: Dict[str, Dict[str, Any]] = {}
    for _, row in work.iterrows():
        subject = str(row[subject_col])
        task = str(row.get(task_col, "unknown"))
        phase = str(row.get(phase_col, "unknown"))
        prefix = f"{modality}__{make_row_key(task, phase)}"
        out = rows.setdefault(subject, {"subject": subject})
        for col, value in row.items():
            if col in extra_drop:
                continue
            if col in {subject_col, task_col, phase_col}:
                continue
            if not is_feature_column(work, col):
                continue
            numeric = to_float(value)
            if numeric is None:
                continue
            out[f"{prefix}__{clean_name(col)}"] = numeric
    return pd.DataFrame(list(rows.values()))


def build_interaction_table(records: List[PhaseRecord]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        response_path = rec.phase_dir / "response.json"
        if not response_path.exists():
            continue
        try:
            data = json.loads(response_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row = {
            "subject": rec.subject,
            "folder_subject": rec.folder_subject,
            "subject_code": rec.subject_code,
            "device_id": rec.device_id,
            "version": rec.version,
            "timestamp": rec.timestamp,
            "task": rec.task,
            "phase": rec.phase,
        }
        row.update(flatten_json_numeric(data))
        rows.append(row)
    return pd.DataFrame(rows)


def load_group_info(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["subject"])
    df = pd.read_csv(path, low_memory=False)
    if "subject" not in df.columns:
        raise ValueError(f"group info file has no subject column: {path}")
    df["subject"] = df["subject"].astype(str)
    return df.drop_duplicates(subset=["subject"], keep="first")


def standardize_feature_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rename: Dict[str, str] = {}
    for col in df.columns:
        if col.startswith("interaction__"):
            rename[col] = "inter__" + col[len("interaction__") :]
        elif col.startswith("infer__"):
            rename[col] = "inter__" + col[len("infer__") :]
    if not rename:
        return df
    return df.rename(columns=rename)


def build_base_subjects(records: List[PhaseRecord], group_info: pd.DataFrame) -> pd.DataFrame:
    rows = {
        rec.subject: {
            "subject": rec.subject,
            "subject_code": rec.subject_code,
            "device_id": rec.device_id,
            "source_timestamp": rec.timestamp,
        }
        for rec in records
    }
    base = pd.DataFrame(list(rows.values()))
    if group_info.empty:
        return base
    keep_cols = [c for c in group_info.columns if c != "label"]
    if base.empty:
        return group_info[keep_cols + (["label"] if "label" in group_info.columns else [])].copy()
    merged = base.merge(group_info[keep_cols + (["label"] if "label" in group_info.columns else [])], on="subject", how="left")
    return merged


def encode_group_info_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = standardize_feature_prefixes(out)
    if "inter__formal__Sight" in out.columns:
        sight = out["inter__formal__Sight"].fillna("").astype(str)
        out["inter__formal__Sight_any"] = sight.str.len().gt(0).astype(int)
        out["inter__formal__Sight_has_1a"] = sight.str.contains("1a", regex=False).astype(int)
        out["inter__formal__Sight_has_non_1a"] = sight.map(
            lambda x: int(any(tok and tok != "1a" for tok in [x[i : i + 2] for i in range(0, len(x), 2)]))
        )
    for col in list(out.columns):
        if col in {"subject", "label", "subject_code", "device_id"}:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().sum() > 0:
            out[col] = numeric
            continue
        dummies = pd.get_dummies(out[col].fillna("missing").astype(str), prefix=clean_name(col), dummy_na=False)
        out = pd.concat([out.drop(columns=[col]), dummies.astype(int)], axis=1)
    return out


def filter_long_features(df: pd.DataFrame, tasks: Optional[List[str]], versions: Optional[List[str]]) -> pd.DataFrame:
    out = df.copy()
    if tasks is not None and "task" in out.columns:
        out = out[out["task"].astype(str).isin(set(tasks))].copy()
    if versions:
        version_filter = {v.lower() for v in versions}
        if "version" in out.columns:
            out = out[out["version"].fillna("").astype(str).str.lower().isin(version_filter)].copy()
        elif "subject" in out.columns:
            out = out[
                out["subject"].astype(str).map(lambda x: parse_private_subject(x)[3].lower()).isin(version_filter)
            ].copy()
        elif "folder_subject" in out.columns:
            out = out[
                out["folder_subject"].astype(str).map(lambda x: parse_private_subject(x)[3].lower()).isin(version_filter)
            ].copy()
    return out


def load_gaze_table(gaze_summary: Path, tasks: Optional[List[str]] = None, versions: Optional[List[str]] = None) -> pd.DataFrame:
    if not gaze_summary.exists():
        return pd.DataFrame(columns=["subject"])
    df = pd.read_csv(gaze_summary, low_memory=False)
    df = filter_long_features(df, tasks=tasks, versions=versions)
    if "calibration_key" in df.columns:
        df["subject"] = df["calibration_key"].astype(str)
    elif "subject" in df.columns:
        df["subject"] = df["subject"].astype(str).map(lambda x: parse_private_subject(x)[0])
    return wide_from_long(df, "gaze", "subject", "task", "phase")


def load_audio_table(audio_csv: Path, tasks: Optional[List[str]] = None, versions: Optional[List[str]] = None) -> pd.DataFrame:
    if not audio_csv.exists():
        return pd.DataFrame(columns=["subject"])
    df = pd.read_csv(audio_csv, low_memory=False)
    df = filter_long_features(df, tasks=tasks, versions=versions)
    if "folder_subject" in df.columns:
        df["subject"] = df["folder_subject"].astype(str).map(lambda x: parse_private_subject(x)[0])
    elif "subject" in df.columns:
        df["subject"] = df["subject"].astype(str).map(lambda x: parse_private_subject(x)[0])
    else:
        return pd.DataFrame(columns=["subject"])
    return wide_from_long(df, "audio", "subject", "task", "phase")


def build_multimodal_table(
    data_root: Path,
    output_dir: Path,
    group_info_path: Path,
    gaze_summary: Path,
    audio_csv: Path,
    subjects: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    gaze_tasks: Optional[List[str]] = None,
    audio_tasks: Optional[List[str]] = None,
    versions: Optional[List[str]] = None,
    timestamp: Optional[str] = None,
    assessment_phase: Optional[str] = None,
    include_t001: bool = True,
    logger: Optional[StageLogger] = None,
) -> pd.DataFrame:
    ensure_dir(output_dir)
    with optional_stage(logger, "数据加载"):
        records = discover_phase_records(
            data_root=data_root,
            subjects=subjects,
            tasks=tasks,
            versions=versions,
            include_t001=include_t001,
            timestamp=timestamp,
            assessment_phase=assessment_phase,
        )
        group_info = load_group_info(group_info_path)
        base = build_base_subjects(records, group_info)
        base = encode_group_info_features(base)
        if logger:
            logger.log(f"[DATA] phase_records={len(records)} subjects={base['subject'].nunique() if 'subject' in base.columns else 0}")

    with optional_stage(logger, "模态特征表读取"):
        gaze_wide = load_gaze_table(gaze_summary, tasks=gaze_tasks if gaze_tasks is not None else tasks, versions=versions)
        audio_wide = load_audio_table(audio_csv, tasks=audio_tasks if audio_tasks is not None else tasks, versions=versions)
        if logger:
            logger.log(f"[FEATURES] gaze_shape={gaze_wide.shape} audio_shape={audio_wide.shape}")

    with optional_stage(logger, "Coginter 提取交互特征"):
        interaction_long = build_interaction_table(records)
        interaction_wide = wide_from_long(interaction_long, "inter", "subject", "task", "phase")
        if logger:
            logger.log(f"[FEATURES] interaction_shape={interaction_wide.shape}")

    with optional_stage(logger, "多模态特征融合"):
        merged = base.copy()
        for table in [gaze_wide, audio_wide, interaction_wide]:
            if table.empty or "subject" not in table.columns:
                continue
            merged = merged.merge(table, on="subject", how="left")

        merged = standardize_feature_prefixes(merged)
        merged = merged.drop_duplicates(subset=["subject"], keep="first").reset_index(drop=True)
        out_csv = output_dir / "multimodal_subject_features.csv"
        merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
        manifest = {
            "data_root": str(data_root),
            "group_info_path": str(group_info_path),
            "gaze_summary": str(gaze_summary),
            "audio_csv": str(audio_csv),
            "subjects": subjects,
            "tasks": tasks,
            "gaze_tasks": gaze_tasks,
            "audio_tasks": audio_tasks,
            "versions": versions,
            "timestamp": timestamp,
            "assessment_phase": assessment_phase,
            "include_t001": include_t001,
            "n_subjects": int(len(merged)),
            "n_columns": int(len(merged.columns)),
        }
        (output_dir / "feature_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if logger:
            logger.log(f"[FUSION] saved={out_csv} shape={merged.shape}")
    return merged
