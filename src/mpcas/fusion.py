"""Fuse long-form modality summaries into task tables for model preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import _protected_feature_name, read_metadata


IDENTIFIER_COLUMNS = {
    "subject", "task", "phase", "record_key", "folder_subject", "inner_subject",
    "subject_code", "device_id", "device", "version", "timestamp", "source_root",
    "video_path", "source_video", "wav_path", "file_path", "subject_raw",
    "subject_bucket", "subject_dir_name", "calibration_key", "orientation_used",
    "status", "error", "audio_extract_ok", "fps_original", "frames_total",
    "frames_sampled", "frames_valid_face", "target_fps", "label_x", "label_y",
    "raw_pred_x", "raw_pred_y", "is_calibration", "audio_extract_ok",
}


def _one_modality(path: Path, modality: str, subjects: set[str], tasks: list[str]) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"subject", "task", "phase"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks {sorted(required - set(frame.columns))}")
    for field in required:
        frame[field] = frame[field].astype("string").str.strip()
    if frame[["subject", "task", "phase"]].isna().any().any():
        raise ValueError(f"Missing subject/task/phase in {path}")
    frame = frame[frame.subject.isin(subjects) & frame.task.isin(tasks)].copy()
    frame = frame[~frame.phase.str.lower().isin(["formalitem", "practiceitem"])].copy()
    if frame.duplicated(["subject", "task", "phase"]).any():
        raise ValueError(f"Duplicate subject/task/phase records in {path}; select one assessment session")
    feature_columns = [
        col for col in frame if col.lower() not in IDENTIFIER_COLUMNS
        and not _protected_feature_name(col)
    ]
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    feature_columns = [col for col in feature_columns if numeric[col].notna().any()]
    if not feature_columns:
        raise ValueError(f"No numeric features in {path}")
    frame[feature_columns] = numeric[feature_columns]
    by_task = {}
    for task in tasks:
        rows = frame.loc[frame.task.eq(task), ["subject", "phase", *feature_columns]]
        if rows.empty:
            continue
        wide = rows.set_index(["subject", "phase"])[feature_columns].unstack("phase")
        wide.columns = [f"{modality}__{task}__{phase}__{feature}" for feature, phase in wide.columns]
        by_task[task] = wide.reset_index()
    return by_task


def fuse_modalities(metadata_path: Path, output_dir: Path, tasks: list[str], inputs: dict[str, Path]) -> None:
    if not inputs:
        raise ValueError("At least one modality CSV is required")
    metadata = read_metadata(metadata_path)
    subjects = metadata.subject.astype(str).str.strip()
    if subjects.duplicated().any():
        raise ValueError("Duplicate subjects in metadata")
    subject_set = set(subjects)
    parts = {name: _one_modality(path, name, subject_set, tasks) for name, path in inputs.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        merged = pd.DataFrame({"subject": subjects})
        for modality in ["screen-interaction", "gaze", "facial", "audio-language"]:
            part = parts.get(modality, {}).get(task)
            if part is not None:
                merged = merged.merge(part, on="subject", how="left", validate="one_to_one")
        if len(merged.columns) == 1:
            raise ValueError(f"No modality features found for {task}")
        merged.to_csv(output_dir / f"{task}.csv", index=False)
