"""Modular gaze feature extraction component."""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from mpcas_gaze.device import VIEWING_DISTANCE_CM, default_device_for_subject, parse_device_info
from mpcas_gaze.pathing import (
    SubjectParts, choose_rawdata_timestamp, is_rawdata_root, is_timestamp_name,
    parse_subject_folder, parse_timestamp, rawdata_timestamps,
    subject_filter_matches, version_filter_matches,
)
from .gaze_core import *
from .gaze_core import _maybe_scale_time_to_ms


def collect_fixation_calibration_points(
    gaze_df: pd.DataFrame,
    dotinfo: List[dict],
    device_meta: Dict[str, object],
    max_points: int = 5,
    phone_max_error_cm: float = 4.0,
    tablet_max_error_cm: float = 8.0,
    settle_ms: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray, int, int, int, int, float]:
    if not dotinfo or gaze_df.empty:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), 0, 0, 0, 0, float("nan")

    g = gaze_df.copy()
    g["time_ms"] = pd.to_numeric(g["time_ms"], errors="coerce")
    g["x_cam"] = pd.to_numeric(g["x_cam"], errors="coerce")
    g["y_cam"] = pd.to_numeric(g["y_cam"], errors="coerce")
    g = g.loc[robust_valid_mask(g)]
    if g.empty:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), 0, 0, 0, 0, float("nan")

    device_class = str(device_meta.get("device_class", "unknown")).lower()
    if device_class == "phone":
        max_error_cm = float(phone_max_error_cm)
    elif device_class == "tablet":
        max_error_cm = float(tablet_max_error_cm)
    else:
        max_error_cm = float(max(phone_max_error_cm, tablet_max_error_cm))

    # Use the first N unique calibration points. T017 is a 5-point task;
    # T002 has 13 points but the first five match the 5-point calibration
    # protocol requested here.
    selected_events: List[dict] = []
    seen_points = set()
    indexed_dotinfo = list(enumerate(dotinfo))
    for _original_idx, event in sorted(
        indexed_dotinfo,
        key=lambda pair: (
            first_finite_value(pair[1], ("dot_num",), float(pair[0])),
            first_finite_value(pair[1], ("time_offset",), 0.0),
            pair[0],
        ),
    ):
        tx = first_finite_value(event, ("x_cam",))
        ty = first_finite_value(event, ("y_cam",))
        if not (np.isfinite(tx) and np.isfinite(ty)):
            continue
        dot_key = event.get("dot_num")
        if dot_key is None:
            dot_key = (round(float(tx), 4), round(float(ty), 4))
        if dot_key in seen_points:
            continue
        seen_points.add(dot_key)
        selected_events.append(event)
        if len(selected_events) >= max_points:
            break

    gaze_time = g["time_ms"].values.astype(float)
    starts = np.array([first_finite_value(d, ("time_offset", "half_second_time"), 0.0) for d in selected_events], dtype=float)
    stable_starts = np.array([first_finite_value(d, ("half_second_time",)) for d in selected_events], dtype=float)
    ends = np.array(
        [first_finite_value(d, ("disappear_time", "disappear_time_offset", "pulse_end_time")) for d in selected_events],
        dtype=float,
    )
    durations = np.array([first_finite_value(d, ("duration",)) for d in selected_events], dtype=float)

    starts, _ = _maybe_scale_time_to_ms(starts, gaze_time)
    stable_starts, _ = _maybe_scale_time_to_ms(stable_starts, gaze_time)
    ends, _ = _maybe_scale_time_to_ms(ends, gaze_time)
    durations, _ = _maybe_scale_time_to_ms(durations, gaze_time)

    missing_end = ~np.isfinite(ends)
    ends[missing_end] = starts[missing_end] + fill_nan(durations[missing_end], 2500.0)
    missing_stable = ~np.isfinite(stable_starts)
    stable_starts[missing_stable] = starts[missing_stable] + settle_ms
    stable_starts = np.minimum(stable_starts, ends)

    pred_points: List[np.ndarray] = []
    target_points: List[np.ndarray] = []
    used_point_indices = set()
    support_samples = 0
    candidate_samples = 0
    rejected_samples = 0

    for idx, event in enumerate(selected_events):
        tx = first_finite_value(event, ("x_cam",))
        ty = first_finite_value(event, ("y_cam",))
        if not (np.isfinite(tx) and np.isfinite(ty)):
            continue

        t0 = stable_starts[idx]
        t1 = ends[idx]
        if not (np.isfinite(t0) and np.isfinite(t1)):
            continue
        if t1 < t0:
            t0, t1 = t1, t0

        window = g[(g["time_ms"] >= t0) & (g["time_ms"] <= t1)]
        if window.empty:
            window = g[(g["time_ms"] >= starts[idx]) & (g["time_ms"] <= t1 + 250.0)]
        if window.empty:
            continue

        coords = window[["x_cam", "y_cam"]].values.astype(float)
        finite = np.isfinite(coords).all(axis=1)
        target = np.array([tx, ty], dtype=float)
        errors = np.linalg.norm(coords - target.reshape(1, 2), axis=1)
        keep = finite & np.isfinite(errors) & (errors <= max_error_cm)
        candidate_samples += int(len(coords))
        rejected_samples += int(np.sum(finite & ~keep))
        if not np.any(keep):
            continue

        kept = coords[keep]
        pred_points.append(kept)
        target_points.append(np.repeat(target.reshape(1, 2), kept.shape[0], axis=0))
        support_samples += int(kept.shape[0])
        used_point_indices.add(idx)

    if not pred_points:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            0,
            support_samples,
            candidate_samples,
            rejected_samples,
            max_error_cm,
        )
    return (
        np.vstack(pred_points).astype(np.float32),
        np.vstack(target_points).astype(np.float32),
        int(len(used_point_indices)),
        support_samples,
        candidate_samples,
        rejected_samples,
        max_error_cm,
    )


def fit_calibration_model(
    key: str,
    source: RecordPaths,
    pred_points: np.ndarray,
    target_points: np.ndarray,
    support_points: int,
    support_samples: int,
    candidate_samples: int,
    rejected_samples: int,
    max_error_cm: float,
    mode: str,
    min_points: int,
) -> Optional[CalibrationModel]:
    mode = (mode or "none").lower()
    if mode == "none" or support_points < min_points or len(pred_points) == 0:
        return None

    if mode == "affine" and support_points >= 3 and len(pred_points) >= 3:
        src = np.hstack([pred_points, np.ones((pred_points.shape[0], 1), dtype=np.float32)])
        matrix, _, _, _ = np.linalg.lstsq(src, target_points, rcond=None)
        return CalibrationModel(
            key=key,
            source_subject=source.subject,
            source_task=source.task,
            source_phase=source.phase,
            mode="affine",
            support_points=int(support_points),
            support_samples=int(support_samples),
            matrix=matrix.astype(np.float32),
        )

    offset = np.nanmean(target_points - pred_points, axis=0).astype(np.float32)
    return CalibrationModel(
        key=key,
        source_subject=source.subject,
        source_task=source.task,
        source_phase=source.phase,
        mode="offset" if mode in {"offset", "affine"} else mode,
        support_points=int(support_points),
        support_samples=int(support_samples),
        offset=offset,
    )


def build_calibration_models(
    records: List[RecordPaths],
    calibration_tasks: List[str],
    calibration_phase: Optional[str],
    mode: str,
    min_points: int,
    max_points: int,
    phone_max_error_cm: float,
    tablet_max_error_cm: float,
) -> Tuple[Dict[str, CalibrationModel], Dict[str, dict]]:
    models: Dict[str, CalibrationModel] = {}
    report: Dict[str, dict] = {}
    if (mode or "none").lower() == "none":
        return models, report

    task_order = calibration_tasks or ["T017", "T002"]
    task_rank = {task: rank for rank, task in enumerate(task_order)}
    by_key: Dict[str, List[RecordPaths]] = {}
    for paths in records:
        if paths.task not in task_rank:
            continue
        if calibration_phase and paths.phase != calibration_phase:
            continue
        if not paths.dotinfo_json:
            continue
        key = parse_subject_folder(paths.subject).calibration_key
        by_key.setdefault(key, []).append(paths)

    for key, key_records in by_key.items():
        key_records = sorted(key_records, key=lambda item: (task_rank.get(item.task, 999), item.phase, item.subject))
        source_reports: List[dict] = []
        source_errors: List[dict] = []
        pred_batches: List[np.ndarray] = []
        target_batches: List[np.ndarray] = []
        total_support_samples = 0
        total_candidate_samples = 0
        total_rejected_samples = 0
        source_paths: List[RecordPaths] = []
        source_device_classes = set()
        source_max_errors: List[float] = []

        for paths in key_records:
            parts = parse_subject_folder(paths.subject)
            default_device = default_device_for_subject(parts.subject_code)
            if paths.info_json:
                device_meta = parse_device_info(load_json(paths.info_json))
            elif default_device:
                device_meta = default_device
            else:
                device_meta = parse_device_info(None)

            pred_points = np.empty((0, 2), dtype=np.float32)
            target_points = np.empty((0, 2), dtype=np.float32)
            support_points = 0
            support_samples = 0
            candidate_samples = 0
            rejected_samples = 0
            max_error_cm = float("nan")
            try:
                gaze_df, _meta = load_gaze_record(paths.gaze_csv, paths.meta_json)
                gaze_df, _flipped_y = harmonize_gaze_coordinates(gaze_df, paths.dotinfo_json)
                dotinfo = load_json(paths.dotinfo_json)
                (
                    pred_points,
                    target_points,
                    support_points,
                    support_samples,
                    candidate_samples,
                    rejected_samples,
                    max_error_cm,
                ) = collect_fixation_calibration_points(
                    gaze_df,
                    dotinfo,
                    device_meta=device_meta,
                    max_points=max_points,
                    phone_max_error_cm=phone_max_error_cm,
                    tablet_max_error_cm=tablet_max_error_cm,
                )
            except Exception as exc:
                source_errors.append(
                    {
                        "source_subject": paths.subject,
                        "source_task": paths.task,
                        "source_phase": paths.phase,
                        "message": str(exc),
                    }
                )
                continue

            source_report = {
                "source_subject": paths.subject,
                "source_task": paths.task,
                "source_phase": paths.phase,
                "support_points": int(support_points),
                "support_samples": int(support_samples),
                "candidate_samples": int(candidate_samples),
                "rejected_samples": int(rejected_samples),
                "requested_points": int(max_points),
                "max_error_cm": float(max_error_cm) if np.isfinite(max_error_cm) else None,
                "device_class": device_meta.get("device_class"),
            }
            source_reports.append(source_report)
            total_support_samples += int(support_samples)
            total_candidate_samples += int(candidate_samples)
            total_rejected_samples += int(rejected_samples)
            if np.isfinite(max_error_cm):
                source_max_errors.append(float(max_error_cm))
            if device_meta.get("device_class"):
                source_device_classes.add(str(device_meta.get("device_class")))

            if len(pred_points) and len(target_points):
                pred_batches.append(pred_points)
                target_batches.append(target_points)
                source_paths.append(paths)

        if not pred_batches:
            report[key] = {
                "status": "skipped",
                "support_points": 0,
                "support_samples": int(total_support_samples),
                "candidate_samples": int(total_candidate_samples),
                "rejected_samples": int(total_rejected_samples),
                "requested_points": int(max_points),
                "required_points": int(min_points),
                "sources": source_reports,
                "errors": source_errors,
            }
            continue

        pred_points_all = np.vstack(pred_batches).astype(np.float32)
        target_points_all = np.vstack(target_batches).astype(np.float32)
        unique_targets = {
            tuple(np.round(row.astype(float), 4))
            for row in target_points_all
            if np.isfinite(row).all()
        }
        support_points_all = int(len(unique_targets))
        source = source_paths[0]
        source_tasks = []
        source_phases = []
        source_subjects = []
        for item in source_paths:
            if item.task not in source_tasks:
                source_tasks.append(item.task)
            if item.phase not in source_phases:
                source_phases.append(item.phase)
            if item.subject not in source_subjects:
                source_subjects.append(item.subject)

        model = fit_calibration_model(
            key=key,
            source=source,
            pred_points=pred_points_all,
            target_points=target_points_all,
            support_points=support_points_all,
            support_samples=total_support_samples,
            candidate_samples=total_candidate_samples,
            rejected_samples=total_rejected_samples,
            max_error_cm=max(source_max_errors) if source_max_errors else float("nan"),
            mode=mode,
            min_points=min_points,
        )

        common_report = {
            "source_subject": ",".join(source_subjects),
            "source_task": ",".join(source_tasks),
            "source_phase": ",".join(source_phases),
            "support_points": int(support_points_all),
            "support_samples": int(total_support_samples),
            "candidate_samples": int(total_candidate_samples),
            "rejected_samples": int(total_rejected_samples),
            "requested_points": int(max_points),
            "required_points": int(min_points),
            "max_error_cm": max(source_max_errors) if source_max_errors else None,
            "device_class": ",".join(sorted(source_device_classes)) if source_device_classes else None,
            "sources": source_reports,
            "errors": source_errors,
        }
        if model is None:
            report[key] = {"status": "skipped", **common_report}
            continue

        model.source_subject = common_report["source_subject"]
        model.source_task = common_report["source_task"]
        model.source_phase = common_report["source_phase"]
        models[key] = model
        report[key] = {
            "status": "ok",
            **common_report,
            "mode": model.mode,
            "offset": model.offset.tolist() if model.offset is not None else None,
        }

    return models, report


def apply_calibration_to_gaze(gaze_df: pd.DataFrame, model: Optional[CalibrationModel]) -> pd.DataFrame:
    if model is None or gaze_df.empty:
        return gaze_df
    out = gaze_df.copy()
    coords = out[["x_cam", "y_cam"]].values.astype(float)
    mask = np.isfinite(coords).all(axis=1)
    if not np.any(mask):
        return out
    calibrated = model.predict(coords[mask]).astype(float)
    out.loc[mask, "x_cam"] = calibrated[:, 0]
    out.loc[mask, "y_cam"] = calibrated[:, 1]
    return out
