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
from .gaze_targets import *
from .gaze_calibration import *


def iter_records(
    gaze_root: Path,
    data_root: Path,
    subjects: Optional[List[str]],
    tasks: Optional[List[str]],
    versions: Optional[List[str]] = None,
    phases: Optional[List[str]] = None,
    timestamp: Optional[str] = None,
    assessment_phase: Optional[str] = None,
) -> Iterable[RecordPaths]:
    if not gaze_root.exists():
        return
    phase_filter = {str(item).strip().lower() for item in phases or [] if str(item).strip()}

    def resolve_data_phase(subject_name: str, task_name: str, phase_name: str) -> Path:
        flat = data_root / subject_name / task_name / phase_name
        if flat.exists():
            return flat
        subject_root = data_root / subject_name
        if subject_root.exists():
            candidates = []
            for session_dir in sorted(subject_root.iterdir()):
                if not session_dir.is_dir():
                    continue
                task_dir = session_dir / task_name
                phase_dir = task_dir / phase_name
                if phase_dir.exists():
                    candidates.append(phase_dir)
                elif task_dir.exists():
                    for child_phase in sorted(task_dir.iterdir()):
                        if child_phase.is_dir():
                            candidates.append(child_phase)
            candidates = sorted(
                candidates,
                key=lambda path: (0 if str(path).lower().find("_advanced") >= 0 else 1, str(path)),
            )
            for candidate in candidates:
                if candidate.name == phase_name:
                    return candidate
            for candidate in candidates:
                dot_path = candidate / "dotInfo.json"
                if not dot_path.exists():
                    continue
                try:
                    rows = load_json(dot_path)
                except Exception:
                    continue
                if any(str(row.get("phase", "")) == phase_name for row in rows if isinstance(row, dict)):
                    return candidate
            if candidates:
                return candidates[0]
        return flat

    for subj_dir in gaze_root.iterdir():
        if not subj_dir.is_dir():
            continue
        if not subject_filter_matches(subj_dir.name, subjects):
            continue
        if not version_filter_matches(subj_dir.name, versions):
            continue
        for task_dir in subj_dir.iterdir():
            if not task_dir.is_dir():
                continue
            if tasks and task_dir.name not in tasks:
                continue
            for phase_dir in task_dir.iterdir():
                if not phase_dir.is_dir():
                    continue
                phase_name = phase_dir.name.lower()
                if phase_name in EXCLUDED_PHASES:
                    continue
                if phase_filter and phase_name not in phase_filter:
                    continue
                gaze_csv = phase_dir / "video" / "gaze_pred.csv"
                meta_json = phase_dir / "video" / "gaze_mediapipe_meta.json"
                if not (gaze_csv.exists() and meta_json.exists()):
                    gaze_csv = phase_dir / "gaze_pred.csv"
                    meta_json = phase_dir / "gaze_mediapipe_meta.json"
                if not (gaze_csv.exists() and meta_json.exists()):
                    continue
                parts = parse_subject_folder(subj_dir.name)
                if is_rawdata_root(data_root) and parts.version:
                    subject_folder = data_root / parts.calibration_key
                    selected_timestamp = choose_rawdata_timestamp(subject_folder, parts.version, timestamp, assessment_phase)
                    data_phase = (
                        subject_folder / parts.version / selected_timestamp / task_dir.name / phase_dir.name
                        if selected_timestamp
                        else data_root / subj_dir.name / task_dir.name / phase_dir.name
                    )
                else:
                    data_phase = resolve_data_phase(subj_dir.name, task_dir.name, phase_dir.name)
                info_json = data_phase / "info.json"
                dotinfo_json = data_phase / "dotInfo.json"
                response_json = data_phase / "response.json"
                yield RecordPaths(
                    subject=subj_dir.name,
                    task=task_dir.name,
                    phase=phase_dir.name,
                    gaze_csv=gaze_csv,
                    meta_json=meta_json,
                    info_json=info_json if info_json.exists() else None,
                    dotinfo_json=dotinfo_json if dotinfo_json.exists() else None,
                    response_json=response_json if response_json.exists() else None,
                )


def flatten_features(prefix: str, feat: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in feat.items()}


def extract_record_features(
    paths: RecordPaths,
    calibration_models: Optional[Dict[str, CalibrationModel]] = None,
    calibration_tasks: Optional[List[str]] = None,
    apply_calibration_to_support: bool = True,
) -> Tuple[dict, dict]:
    gaze_df, _meta = load_gaze_record(paths.gaze_csv, paths.meta_json)
    gaze_df, flipped_y = harmonize_gaze_coordinates(gaze_df, paths.dotinfo_json)

    info: Optional[dict] = None
    parts = parse_subject_folder(paths.subject)
    base = {
        "subject": paths.subject,
        "subject_code": parts.subject_code,
        "device_id": parts.device_id,
        "version": parts.version,
        "calibration_key": parts.calibration_key,
        "task": paths.task,
        "phase": paths.phase,
    }

    default_device = default_device_for_subject(parts.subject_code)
    if paths.info_json:
        info = load_json(paths.info_json)
        device_meta = parse_device_info(info)
    elif default_device:
        device_meta = default_device
    else:
        device_meta = parse_device_info(None)

    base.update(device_meta)
    base.pop("device_model", None)

    calibration_model = None
    support_tasks = set(calibration_tasks or [])
    if calibration_models:
        if paths.task not in support_tasks or apply_calibration_to_support:
            calibration_model = calibration_models.get(parts.calibration_key)
            gaze_df = apply_calibration_to_gaze(gaze_df, calibration_model)

    gaze_feat = summarize_gaze(gaze_df, norm_scale_cm=device_meta.get("screen_diag_cm"))
    trajectory_features = summarize_gaze_trajectory(gaze_df, device_meta)
    target_features = compute_target_tracking_metrics(gaze_df, paths.dotinfo_json, device_meta)

    detail = {
        "path_info": {
            "gaze_csv": str(paths.gaze_csv),
            "meta_json": str(paths.meta_json),
            "dotinfo_json": str(paths.dotinfo_json) if paths.dotinfo_json else None,
        },
        "preprocess": {"y_cam_flipped": flipped_y},
        "calibration": {
            "applied": calibration_model is not None,
            "key": parts.calibration_key,
            "source_subject": calibration_model.source_subject if calibration_model else None,
            "source_task": calibration_model.source_task if calibration_model else None,
            "source_phase": calibration_model.source_phase if calibration_model else None,
            "mode": calibration_model.mode if calibration_model else None,
            "support_points": calibration_model.support_points if calibration_model else 0,
            "support_samples": calibration_model.support_samples if calibration_model else 0,
        },
    }

    summary = {
        **base,
        **flatten_features("gaze_", gaze_feat),
        **flatten_features("gaze_trajectory_", trajectory_features),
        **flatten_features("target_tracking_", target_features),
    }
    return summary, detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract task-level eye-movement features from predicted gaze.")
    parser.add_argument("--gaze-root", type=Path, default=Path("Results/gaze_pred"))
    parser.add_argument("--data-root", type=Path, default=Path("Rawdata"))
    parser.add_argument("--output", type=Path, default=Path("outputs/gaze"))
    parser.add_argument("--subjects", nargs="*", help="Optional subject/device/folder filter (e.g., E0019)")
    parser.add_argument("--tasks", nargs="*", help="Optional task filter (e.g., T004 T006 ...)")
    parser.add_argument("--versions", nargs="*", choices=["self", "advanced"], help="Optional version filter.")
    parser.add_argument("--phases", nargs="*", default=None, help="Optional phase/stage filter. formal includes T007 formalcolor; practice includes T007 practicecolor.")
    parser.add_argument("--timestamp", type=str, default=None, help="Rawdata assessment timestamp, e.g. 20260427T153045.")
    parser.add_argument("--assessment-phase", choices=["self", "advanced"], default=None, help="Assessment phase used when selecting Rawdata timestamps.")
    parser.add_argument(
        "--calibration-tasks",
        default="T017,T002",
        help="Comma-separated calibration source tasks. Available T017 and first 5 points of T002 are pooled by default.",
    )
    parser.add_argument("--calibration-phase", default="formal", help="Phase used for calibration; use empty string for any phase.")
    parser.add_argument("--calibration-mode", default="offset", choices=["none", "offset", "affine"])
    parser.add_argument("--calibration-points", type=int, default=5, help="Number of initial calibration points to use.")
    parser.add_argument("--calibration-min-points", type=int, default=4, help="Minimum kept calibration points required to fit a model.")
    parser.add_argument("--calibration-phone-max-error-cm", type=float, default=4.0)
    parser.add_argument("--calibration-tablet-max-error-cm", type=float, default=8.0)
    parser.add_argument(
        "--calibrate-support-task",
        dest="calibrate_support_task",
        action="store_true",
        default=True,
        help="Also apply the fitted calibration to calibration source tasks when extracting features.",
    )
    parser.add_argument(
        "--no-calibrate-support-task",
        dest="calibrate_support_task",
        action="store_false",
        help="Do not apply calibration to calibration source tasks themselves.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Write only gaze_features.csv; skip per-record JSON and calibration_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = resolve_existing_path(args.data_root, fallback_names=("Rawdata",))
    args.gaze_root = resolve_existing_path(args.gaze_root)
    ensure_dir(args.output)
    output_phases = resolve_phase_filter(args.phases)
    calibration_phase = args.calibration_phase or None
    calibration_tasks = parse_csv_arg(args.calibration_tasks)

    records = list(
        iter_records(
            args.gaze_root,
            args.data_root,
            args.subjects,
            args.tasks,
            args.versions,
            phases=output_phases,
            timestamp=args.timestamp,
            assessment_phase=args.assessment_phase,
        )
    )
    calibration_records = records
    if args.calibration_mode != "none" and calibration_tasks:
        calibration_phases = [calibration_phase] if calibration_phase else output_phases
        calibration_records = list(
            iter_records(
                args.gaze_root,
                args.data_root,
                args.subjects,
                calibration_tasks,
                args.versions,
                phases=calibration_phases,
                timestamp=args.timestamp,
                assessment_phase=args.assessment_phase,
            )
        )
    calibration_models, calibration_report = build_calibration_models(
        calibration_records,
        calibration_tasks=calibration_tasks,
        calibration_phase=calibration_phase,
        mode=args.calibration_mode,
        min_points=args.calibration_min_points,
        max_points=args.calibration_points,
        phone_max_error_cm=args.calibration_phone_max_error_cm,
        tablet_max_error_cm=args.calibration_tablet_max_error_cm,
    )
    calibration_report["_settings"] = {
        "calibration_tasks": calibration_tasks,
        "calibration_phase": calibration_phase,
        "calibration_mode": args.calibration_mode,
        "calibration_points": int(args.calibration_points),
        "calibration_min_points": int(args.calibration_min_points),
        "calibration_phone_max_error_cm": float(args.calibration_phone_max_error_cm),
        "calibration_tablet_max_error_cm": float(args.calibration_tablet_max_error_cm),
        "calibrate_support_task": bool(args.calibrate_support_task),
    }

    summaries: List[dict] = []
    for paths in records:
        summary, detail = extract_record_features(
            paths,
            calibration_models=calibration_models,
            calibration_tasks=calibration_tasks,
            apply_calibration_to_support=args.calibrate_support_task,
        )
        summaries.append(summary)

        if not args.csv_only:
            out_dir = args.output / paths.subject / paths.task
            ensure_dir(out_dir)
            out_path = out_dir / f"{paths.phase}_features.json"
            payload = {"summary": summary, "detail": detail}
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if summaries:
        df = pd.DataFrame(summaries)
        csv_path = args.output / "gaze_features.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(summaries)} summaries to {csv_path}")
        if not args.csv_only:
            report_path = args.output / "calibration_report.json"
            report_path.write_text(json.dumps(calibration_report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved calibration report to {report_path}")
    else:
        print("No gaze predictions found under the provided root.")
