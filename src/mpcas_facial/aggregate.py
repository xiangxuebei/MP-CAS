"""Aggregate OpenFace AU and ResMaskNet emotion frames into task descriptors."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


AU_NAMES = {
    1: "inner_brow_raiser", 2: "outer_brow_raiser", 4: "brow_lowerer",
    5: "upper_lid_raiser", 6: "cheek_raiser", 7: "lid_tightener",
    9: "nose_wrinkler", 10: "upper_lip_raiser", 12: "lip_corner_puller",
    14: "dimpler", 15: "lip_corner_depressor", 17: "chin_raiser",
    20: "lip_stretcher", 23: "lip_tightener", 24: "lip_pressor",
    25: "lips_part", 26: "jaw_drop",
}
EMOTIONS = ("neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger", "contempt")
REQUIRED_SIGNALS = {*(f"au_{number}_intensity" for number in (1, 2, 4, 5, 6, 7, 10, 12, 14, 15, 17, 20, 23, 25, 26)),
                  *(f"emotion_{name}" for name in ("neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger")),
                  "head_yaw", "head_pitch", "head_roll"}


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lookup = {str(column).lower(): column for column in frame.columns}
    return next((lookup[name.lower()] for name in names if name.lower() in lookup), None)


def _normalise_frames(frame: pd.DataFrame, fps: float) -> pd.DataFrame:
    rows = frame.copy()
    index_col = _first_column(rows, ("frame_idx", "frame_index"))
    indices = pd.to_numeric(rows[index_col], errors="coerce") if index_col else pd.Series(np.arange(len(rows)), index=rows.index)
    if indices.isna().any() or indices.duplicated().any():
        raise ValueError("Frame indices must be numeric and unique")
    rows["time_s"] = indices.to_numpy(dtype=float) / fps
    rows = rows.sort_values("time_s").reset_index(drop=True)
    for number, label in AU_NAMES.items():
        source = _first_column(rows, (f"au_{number}_intensity", f"au{number:02d}_intensity", f"AU{number:02d}_r"))
        if source is None:
            continue
        values = pd.to_numeric(rows[source], errors="coerce")
        if values.dropna().empty:
            continue
        # OpenFace reports 0–5; task statistics and activation thresholds use 0–1.
        if values.min() < 0 or values.max() > 5:
            raise ValueError(f"Invalid OpenFace AU intensity range: {source}")
        rows[f"au{number:02d}_{label}"] = values / 5.0
    for emotion in EMOTIONS:
        source = _first_column(rows, (f"emotion_{emotion}", f"{emotion}_probability", f"prob_{emotion}"))
        if source is not None:
            values = pd.to_numeric(rows[source], errors="coerce")
            if (values.dropna() < 0).any() or (values.dropna() > 1).any():
                raise ValueError(f"Emotion probabilities must lie within 0–1: {source}")
            rows[f"emotion_{emotion}"] = values
    for axis in ("yaw", "pitch", "roll"):
        source = _first_column(rows, (axis, f"head_{axis}"))
        if source is not None:
            rows[f"head_{axis}"] = pd.to_numeric(rows[source], errors="coerce")
    return rows


def missing_required_signals(frame: pd.DataFrame) -> list[str]:
    missing: list[str] = []
    for source in sorted(REQUIRED_SIGNALS):
        au = re.fullmatch(r"au_(\d+)_intensity", source)
        if au:
            number = int(au.group(1))
            names = (source, f"au{number:02d}_intensity", f"AU{number:02d}_r")
        elif source.startswith("emotion_"):
            emotion = source.removeprefix("emotion_")
            names = (source, f"{emotion}_probability", f"prob_{emotion}")
        else:
            names = (source, source.removeprefix("head_"))
        column = _first_column(frame, names)
        if column is None or not pd.to_numeric(frame[column], errors="coerce").notna().any():
            missing.append(source)
    return missing


def _episodes(time: np.ndarray, active: np.ndarray, fps: float) -> tuple[int, float, float]:
    contiguous = np.diff(time) <= 1.5 / fps
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1] & contiguous])
    ends = np.flatnonzero(active & ~np.r_[active[1:] & contiguous, False])
    if not len(starts):
        return 0, 0.0, 0.0
    durations = (ends - starts + 1) / fps
    return len(starts), float(np.mean(durations)), float(np.max(durations))


def _stats(prefix: str, time: np.ndarray, values: np.ndarray, fps: float, is_au: bool,
           activity: bool = True) -> dict[str, float]:
    valid = np.isfinite(values)
    finite = values[valid]
    result: dict[str, float] = {}
    if not len(finite):
        return result
    result.update({f"{prefix}_mean": float(np.mean(finite)), f"{prefix}_median": float(np.median(finite)),
                   f"{prefix}_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                   f"{prefix}_min": float(np.min(finite)), f"{prefix}_max": float(np.max(finite)),
                   f"{prefix}_p10": float(np.percentile(finite, 10)), f"{prefix}_p90": float(np.percentile(finite, 90))})
    if len(finite) > 1 and np.ptp(time[valid]) > 0:
        result[f"{prefix}_slope_per_s"] = float(np.polyfit(time[valid], finite, 1)[0])
    if not activity:
        return result
    threshold = max(0.55, float(np.percentile(finite, 75))) if is_au else 0.35
    active = valid & (values >= threshold)
    count, mean_duration, max_duration = _episodes(time, active, fps)
    result.update({f"{prefix}_thr": threshold, f"{prefix}_activation_proportion": float(np.mean(active[valid])),
                   f"{prefix}_segments": float(count), f"{prefix}_mean_episode_duration_s": mean_duration,
                   f"{prefix}_max_episode_duration_s": max_duration,
                   f"{prefix}_episode_frequency_per_min": float(count / max((time[-1] - time[0] + 1 / fps) / 60, 1e-9))})
    if is_au:
        peaks, _ = find_peaks(np.where(valid, values, -np.inf), height=threshold)
        result[f"{prefix}_peak_count"] = float(len(peaks))
        result[f"{prefix}_peak_frequency_per_min"] = float(len(peaks) / max((time[-1] - time[0] + 1 / fps) / 60, 1e-9))
    return result


def aggregate_frame_signals(frame: pd.DataFrame, *, fps: float = 30.0) -> dict[str, float]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    rows = _normalise_frames(frame, fps)
    time = rows["time_s"].to_numpy(dtype=float)
    result: dict[str, float] = {"face_valid_frames": float(len(rows)), "original_frame_rate": float(fps)}
    for name in rows.columns:
        if not re.fullmatch(r"au\d{2}_[a-z_]+|emotion_[a-z]+", str(name)):
            continue
        values = pd.to_numeric(rows[name], errors="coerce").to_numpy(dtype=float)
        result.update(_stats(name, time, values, fps, name.startswith("au")))
    for axis in ("yaw", "pitch", "roll"):
        key = f"head_{axis}"
        if key not in rows:
            continue
        values = rows[key].to_numpy(dtype=float)
        result.update(_stats(key, time, values, fps, False, activity=False))
        delta = np.diff(values)
        dt = np.diff(time)
        valid_pair = np.isfinite(values[1:]) & np.isfinite(values[:-1]) & (dt > 0) & (dt <= 1.5 / fps)
        if valid_pair.any():
            velocity = np.abs(delta[valid_pair] / dt[valid_pair])
            result[f"{key}_angular_velocity_mean"] = float(np.mean(velocity))
            result[f"{key}_angular_velocity_median"] = float(np.median(velocity))
            result[f"{key}_angular_velocity_max"] = float(np.max(velocity))
            result[f"{key}_angular_velocity_sd"] = float(np.std(velocity, ddof=1)) if len(velocity) > 1 else np.nan
    positive = rows.get("emotion_happiness")
    negative = [rows.get(f"emotion_{name}") for name in ("sadness", "anger", "fear", "disgust")]
    if positive is not None and all(value is not None for value in negative):
        valence = positive.to_numpy(dtype=float) - np.mean([value.to_numpy(dtype=float) for value in negative], axis=0)
        changes = np.abs(np.diff(valence))
        if len(changes):
            result["emotion_valence_mean_absolute_change"] = float(np.nanmean(changes))
    return result
