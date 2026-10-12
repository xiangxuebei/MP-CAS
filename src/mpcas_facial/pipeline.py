"""Task-video and precomputed-frame entry points for facial features."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from .aggregate import aggregate_frame_signals, missing_required_signals
from .model_outputs import infer_video, read_frame_csv
from .records import discover_videos


def _write(rows: list[dict], report: list[dict], output_dir: Path, *, provenance: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "au_emotion_features.csv", index=False)
    payload = {"provenance": provenance, "records": report}
    (output_dir / "face_signal_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize(frame: pd.DataFrame, *, subject: str, task: str, phase: str, source: str,
               fps: float, strict: bool, source_fps: float | None = None) -> tuple[dict, dict]:
    missing = missing_required_signals(frame)
    if strict and missing:
        raise ValueError(f"Required facial signals missing for {subject}/{task}/{phase}: {', '.join(missing)}")
    features = aggregate_frame_signals(frame, fps=fps)
    features["original_frame_rate"] = float(source_fps if source_fps is not None else fps)
    row = {"subject": subject, "task": task, "phase": phase, **features}
    report = {"subject": subject, "task": task, "phase": phase, "source": source,
              "frames": len(frame), "fps": fps, "original_frame_rate": features["original_frame_rate"],
              "missing_required_signals": missing}
    return row, report


def aggregate_manifest(manifest_path: Path, output_dir: Path, *, fps: float = 30.0,
                       require_complete_signals: bool = True) -> None:
    manifest = pd.read_csv(manifest_path, dtype={"subject": str, "task": str, "phase": str})
    required = {"subject", "task", "phase", "frame_csv"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest lacks {sorted(required - set(manifest.columns))}")
    if manifest.duplicated(["subject", "task", "phase"]).any():
        raise ValueError("Select one assessment per subject/task/phase")
    rows: list[dict] = []
    report: list[dict] = []
    for record in manifest.itertuples(index=False):
        path = Path(record.frame_csv)
        if not path.is_absolute():
            path = manifest_path.parent / path
        frame = read_frame_csv(path)
        item, quality = _summarize(frame, subject=record.subject, task=record.task,
                                   phase=record.phase, source=str(path), fps=fps,
                                   strict=require_complete_signals,
                                   source_fps=getattr(record, "fps_original", None))
        rows.append(item)
        report.append(quality)
    _write(rows, report, output_dir, provenance={"mode": "precomputed_frame_csv", "manifest": str(manifest_path),
                                                 "model_versions": "Record source models and versions with the frame CSVs"})


def infer_task_videos(dataset_root: Path, output_dir: Path, *, tasks: set[str] | None = None,
                      device: str = "cpu", batch_size: int = 64, fps: float = 30.0,
                      openface_exe: Path | None = None,
                      require_complete_signals: bool = True, save_frames: bool = False) -> None:
    import cv2

    videos = discover_videos(dataset_root, tasks)
    if not videos:
        raise FileNotFoundError(f"No task videos found under {dataset_root}")
    keys = [(video.subject, video.task, video.phase) for video in videos]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated subject/task/phase videos; select one assessment session")
    rows: list[dict] = []
    report: list[dict] = []
    for video in videos:
        capture = cv2.VideoCapture(str(video.path))
        try:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if abs(source_fps - fps) > 0.5:
            raise ValueError(f"Expected {fps:g} fps video, got {source_fps:g}: {video.path}")
        frame = infer_video(video.path, openface_exe=openface_exe, device=device, batch_size=batch_size)
        item, quality = _summarize(frame, subject=video.subject, task=video.task,
                                   phase=video.phase, source=f"OpenFace+ResMaskNet:{video.record_key}",
                                   fps=fps, strict=require_complete_signals, source_fps=source_fps)
        rows.append(item)
        report.append(quality)
        if save_frames:
            target = output_dir / "frame_signals" / Path(video.record_key).with_suffix(".csv")
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(target, index=False)
    try:
        pyfeat_version = version("py-feat")
    except PackageNotFoundError:
        pyfeat_version = "unavailable"
    _write(rows, report, output_dir, provenance={"mode": "OpenFace+Py-Feat ResMaskNet",
                                                 "openface_executable": str(openface_exe or "FeatureExtraction on PATH"),
                                                 "py_feat_version": pyfeat_version,
                                                 "target_fps": fps})
