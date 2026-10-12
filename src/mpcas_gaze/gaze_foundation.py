"""Modular gaze feature extraction component."""
from __future__ import annotations
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


EXCLUDED_PHASES = {"formalitem", "practiceitem"}


PHASE_STAGE_ALIASES = {
    "formal": "formal",
    "official": "formal",
    "real": "formal",
    "正式": "formal",
    "practice": "practice",
    "demo": "practice",
    "demonstration": "practice",
    "演示": "practice",
}


STAGE_PHASES = {
    "formal": {"formal", "formalcolor"},
    "practice": {"practice", "practicecolor"},
}


class RecordPaths:
    subject: str
    task: str
    phase: str
    gaze_csv: Path
    meta_json: Path
    info_json: Optional[Path]
    dotinfo_json: Optional[Path]
    response_json: Optional[Path]


class CalibrationModel:
    key: str
    source_subject: str
    source_task: str
    source_phase: str
    mode: str
    support_points: int
    support_samples: int
    matrix: Optional[np.ndarray] = None
    offset: Optional[np.ndarray] = None

    def predict(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        if self.matrix is not None:
            src = np.hstack([coords, np.ones((coords.shape[0], 1), dtype=np.float32)])
            return src @ self.matrix
        if self.offset is not None:
            return coords + self.offset
        return coords


def resolve_phase_filter(values: Optional[Iterable[str]]) -> List[str]:
    tokens: List[str] = []
    for value in values or []:
        tokens.extend(item.strip() for item in str(value).split(",") if item.strip())
    if not tokens or any(token.lower() in {"all", "*"} for token in tokens):
        phases = set().union(*STAGE_PHASES.values())
    else:
        phases = set()
        for token in tokens:
            key = token.lower()
            stage = PHASE_STAGE_ALIASES.get(key)
            if stage:
                phases.update(STAGE_PHASES[stage])
            elif key not in EXCLUDED_PHASES:
                phases.add(key)
    phases.difference_update(EXCLUDED_PHASES)
    return sorted(phases)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_existing_path(path: Path, fallback_names: Iterable[str] = ()) -> Path:
    if path.exists() or path.is_absolute():
        return path

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    candidates = [project_root / path, script_dir / path]
    candidates.extend(Path(name) for name in fallback_names)
    candidates.extend(project_root / name for name in fallback_names)
    candidates.extend(script_dir / name for name in fallback_names)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def estimate_sample_rate(time_ms: np.ndarray) -> float:
    if len(time_ms) < 2:
        return float("nan")
    dt = np.diff(time_ms)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return float("nan")
    return 1000.0 / np.median(dt)


def median_dt_s(time_s: np.ndarray) -> float:
    if len(time_s) < 2:
        return float("nan")
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if len(dt) else float("nan")


def cm_to_deg(cm: np.ndarray, viewing_dist_cm: float) -> np.ndarray:
    # Visual angle: 2*atan(size/(2*dist)) in radians -> degrees.
    # Works for small angles and keeps device differences bounded.
    return np.degrees(2.0 * np.arctan(np.asarray(cm) / (2.0 * viewing_dist_cm)))


def convex_hull_area(points: np.ndarray) -> float:
    """Monotone chain convex hull area (no external deps)."""
    if len(points) < 3:
        return 0.0

    pts = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def lin_drift(time_s: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(time_s) & np.isfinite(values)
    if mask.sum() < 2:
        return float("nan")
    t = time_s[mask]
    v = values[mask]
    if np.allclose(t, t[0]):
        return 0.0
    coefs = np.polyfit(t, v, 1)
    return float(coefs[0])


def merge_meta(meta_path: Path) -> pd.DataFrame:
    meta = load_json(meta_path)
    rows = []
    for frame in meta.get("frames", []):
        rows.append(
            {
                "frame_index": frame.get("frame_index"),
                "time_ms": frame.get("time_ms"),
                "has_face": bool(frame.get("has_face", 0)),
                "face_valid": bool(frame.get("face", {}).get("IsValid", 0)),
                "left_eye_valid": bool(frame.get("left_eye", {}).get("IsValid", 0)),
                "right_eye_valid": bool(frame.get("right_eye", {}).get("IsValid", 0)),
                "face_grid_valid": bool(frame.get("face_grid", {}).get("IsValid", 0)),
            }
        )
    return pd.DataFrame(rows)


def load_gaze_record(gaze_csv: Path, meta_json: Path) -> Tuple[pd.DataFrame, dict]:
    df = pd.read_csv(gaze_csv)
    meta_df = merge_meta(meta_json)
    merged = df.merge(meta_df, on=["frame_index", "time_ms"], how="left")
    merged["time_ms"] = pd.to_numeric(merged["time_ms"], errors="coerce")
    # Ensure numeric gaze coordinates
    merged["x_cam"] = pd.to_numeric(merged.get("x_cam"), errors="coerce")
    merged["y_cam"] = pd.to_numeric(merged.get("y_cam"), errors="coerce")
    merged = merged.sort_values("time_ms").reset_index(drop=True)
    merged["valid"] = merged["valid"].fillna(0).astype(int)
    meta = load_json(meta_json)
    return merged, meta


def harmonize_gaze_coordinates(
    gaze_df: pd.DataFrame, dotinfo_path: Optional[Path]
) -> Tuple[pd.DataFrame, bool]:
    """
    Ensure gaze coordinates follow the dataset convention (y_cam: positive downward).
    Some model outputs are inverted on the Y axis; we flip when medians disagree
    with expected stimulus coordinates.
    """
    flipped = False

    expected_down_sign = 1.0  # dataset docs: y_cam positive when looking down
    if dotinfo_path is not None and dotinfo_path.exists():
        try:
            dot_df = pd.DataFrame(load_json(dotinfo_path))
            dot_y_med = pd.to_numeric(dot_df.get("y_cam"), errors="coerce").median()
            if np.isfinite(dot_y_med) and dot_y_med < 0:
                expected_down_sign = -1.0
        except Exception:
            pass

    gaze_y_med = pd.to_numeric(gaze_df.get("y_cam"), errors="coerce").median()
    if np.isfinite(gaze_y_med) and np.sign(gaze_y_med) != np.sign(expected_down_sign):
        gaze_df = gaze_df.copy()
        gaze_df["y_cam"] = -gaze_df["y_cam"]
        flipped = True

    return gaze_df, flipped


def first_finite_value(data: dict, keys: Iterable[str], default: float = float("nan")) -> float:
    for key in keys:
        value = data.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return default


def fill_nan(values: np.ndarray, replacement: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    arr[~np.isfinite(arr)] = replacement
    return arr


def parse_csv_arg(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def classify_dotinfo(dotinfo: List[dict]) -> str:
    if not dotinfo:
        return "none"
    first = dotinfo[0]
    if any(key in first for key in ("duration", "half_second_time", "disappear_time", "disappear_time_offset", "pulse_end_time")):
        return "events"
    return "timeline"


def _maybe_scale_time_to_ms(times: np.ndarray, gaze_time_ms: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Detect if dotInfo times are in seconds while gaze times are in ms.
    Returns (scaled_times, scale_factor).
    """
    if times.size == 0 or gaze_time_ms.size == 0:
        return times, 1.0
    if not np.isfinite(times).any() or not np.isfinite(gaze_time_ms).any():
        return times, 1.0
    tmax = float(np.nanmax(times))
    gmax = float(np.nanmax(gaze_time_ms))
    # If dot times look like seconds (e.g., 0-60) while gaze is ms (e.g., 0-60000)
    if np.isfinite(tmax) and np.isfinite(gmax) and (tmax < 1e3) and (gmax > 1e4):
        return times * 1000.0, 1000.0
    return times, 1.0


def dynamic_hit_radius_cm(device_meta: Dict[str, object], base_min: float = 1.5) -> float:
    diag = float(device_meta.get("screen_diag_cm", float("nan")))
    if np.isfinite(diag) and diag > 0:
        return float(max(base_min, 0.05 * diag))  # 5% of diagonal, min 1.5cm
    return float(base_min)
