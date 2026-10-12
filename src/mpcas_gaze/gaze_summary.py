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
from .gaze_foundation import *
from .gaze_events import *


def summarize_gaze_trajectory(
    df: pd.DataFrame,
    device_meta: Dict[str, object],
) -> Dict[str, float]:
    """
    Summarize gaze trajectory features across devices and missing samples.
    """
    out: Dict[str, float] = {}
    if df.empty:
        return out

    time_ms = np.asarray(df["time_ms"].values, dtype=float)
    time_s = time_ms / 1000.0

    ok_mask = robust_valid_mask(df)
    out["valid_ratio_strict"] = float(np.mean(ok_mask))
    out["missing_ratio_strict"] = 1.0 - out["valid_ratio_strict"]
    out.update(compute_blink_stats(time_ms, ok_mask))

    if ok_mask.sum() < 5:
        return out

    v_df = df.loc[ok_mask].copy()
    coords_cm = np.asarray(v_df[["x_cam", "y_cam"]].values, dtype=float)
    t_s = np.asarray(v_df["time_ms"].values, dtype=float) / 1000.0

    # Center & diag-normalize (reduces device bias and absolute offset)
    diag = float(device_meta.get("screen_diag_cm", float("nan")))
    if not np.isfinite(diag) or diag <= 0:
        diag = float("nan")
    center = np.nanmedian(coords_cm, axis=0)
    coords_cm_centered = coords_cm - center
    coords_norm = coords_cm_centered / diag if np.isfinite(diag) else coords_cm_centered

    out["center_x_cm"] = float(center[0])
    out["center_y_cm"] = float(center[1])

    # Spatial entropy / complexity
    occ_ent, tr_ent = spatial_entropy(coords_norm, grid=10)
    out["grid_entropy"] = occ_ent
    out["transition_entropy"] = tr_ent

    # Kinematics in degrees
    device_class = str(device_meta.get("device_class", "unknown"))
    dist_cm = VIEWING_DISTANCE_CM.get(device_class, VIEWING_DISTANCE_CM["unknown"])
    coords_deg = cm_to_deg(coords_cm_centered, dist_cm)
    vel_deg_s = compute_velocities(coords_deg, t_s)

    out["speed_mean_deg_s"], out["speed_median_deg_s"], out["speed_p95_deg_s"] = summarize_list(vel_deg_s)
    out["speed_max_deg_s"] = float(np.nanmax(vel_deg_s)) if np.isfinite(np.nanmax(vel_deg_s)) else float("nan")

    # Event detection (I-VT)
    dt = median_dt_s(t_s)
    # Adjust merge gap roughly to one frame at low fps
    merge_gap = float(np.clip(dt * 0.6, 0.02, 0.10)) if np.isfinite(dt) else 0.06
    saccades, fixations = detect_ivt_events(
        coords_deg=coords_deg,
        time_s=t_s,
        vel_deg_s=vel_deg_s,
        sacc_thr=30.0,
        fix_thr=10.0,
        min_fix_dur=0.10,
        min_sacc_dur=0.02,
        merge_gap=merge_gap,
    )

    out["saccade_count"] = float(len(saccades))
    out["fixation_count"] = float(len(fixations))
    dur_s = float((t_s[-1] - t_s[0])) if len(t_s) > 1 else 0.0
    out["saccade_rate_hz"] = float(len(saccades)) / max(dur_s, 1e-3)
    out["fixation_rate_hz"] = float(len(fixations)) / max(dur_s, 1e-3)

    if saccades:
        amps = [s["amplitude_deg"] for s in saccades]
        peaks = [s["peak_speed_deg_s"] for s in saccades]
        durs = [s["duration_s"] for s in saccades]
        out["saccade_amp_mean_deg"], out["saccade_amp_median_deg"], out["saccade_amp_p95_deg"] = summarize_list(amps)
        out["saccade_peak_mean_deg_s"], out["saccade_peak_median_deg_s"], out["saccade_peak_p95_deg_s"] = summarize_list(peaks)
        out["saccade_dur_mean_s"], out["saccade_dur_median_s"], out["saccade_dur_p95_s"] = summarize_list(durs)
    else:
        out["saccade_amp_mean_deg"] = float("nan")
        out["saccade_peak_mean_deg_s"] = float("nan")
        out["saccade_dur_mean_s"] = float("nan")

    if fixations:
        durs = [f["duration_s"] for f in fixations]
        disps = [f["dispersion_deg"] for f in fixations]
        out["fix_dur_mean_s"], out["fix_dur_median_s"], out["fix_dur_p95_s"] = summarize_list(durs)
        out["fix_disp_mean_deg"], out["fix_disp_median_deg"], out["fix_disp_p95_deg"] = summarize_list(disps)
        out["fix_time_ratio"] = float(np.sum(durs)) / max(dur_s, 1e-3)
    else:
        out["fix_dur_mean_s"] = float("nan")
        out["fix_disp_mean_deg"] = float("nan")
        out["fix_time_ratio"] = 0.0

    return out


def summarize_gaze(df: pd.DataFrame, norm_scale_cm: Optional[float] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if df.empty:
        return out

    time_ms = np.asarray(df["time_ms"].values, dtype=float)
    time_s = time_ms / 1000.0

    # STRICT valid: valid==1 AND finite coords/time
    valid_mask = robust_valid_mask(df)
    valid_df = df.loc[valid_mask]

    out["n_samples"] = float(len(df))
    out["duration_s"] = float((time_ms[-1] - time_ms[0]) / 1000.0) if len(df) > 1 else 0.0
    out["sample_rate_hz"] = estimate_sample_rate(time_ms)

    out["valid_ratio"] = float(np.mean(df["valid"].fillna(0).astype(int).values == 1))
    out["valid_ratio_strict"] = float(np.mean(valid_mask))

    out["has_face_ratio"] = float(df["has_face"].fillna(0).mean())
    out["face_valid_ratio"] = float(df.get("face_valid", 0).fillna(0).mean())
    out["eye_valid_ratio"] = float(
        (df["left_eye_valid"].fillna(0).astype(int) & df["right_eye_valid"].fillna(0).astype(int)).mean()
    )

    if valid_df.empty:
        return out

    v_coords = np.asarray(valid_df[["x_cam", "y_cam"]].values, dtype=float)
    v_time_s = np.asarray(valid_df["time_ms"].values, dtype=float) / 1000.0

    out.update(
        {
            "mean_x_cm": float(np.nanmean(v_coords[:, 0])),
            "mean_y_cm": float(np.nanmean(v_coords[:, 1])),
            "std_x_cm": float(np.nanstd(v_coords[:, 0])),
            "std_y_cm": float(np.nanstd(v_coords[:, 1])),
            "iqr_x_cm": float(np.nanpercentile(v_coords[:, 0], 75) - np.nanpercentile(v_coords[:, 0], 25)),
            "iqr_y_cm": float(np.nanpercentile(v_coords[:, 1], 75) - np.nanpercentile(v_coords[:, 1], 25)),
        }
    )

    bbox_w = float(np.nanmax(v_coords[:, 0]) - np.nanmin(v_coords[:, 0]))
    bbox_h = float(np.nanmax(v_coords[:, 1]) - np.nanmin(v_coords[:, 1]))
    out["bbox_width_cm"] = bbox_w
    out["bbox_height_cm"] = bbox_h
    out["bbox_area_cm2"] = bbox_w * bbox_h

    # Convex hull: drop any remaining NaNs (critical fix)
    finite = np.isfinite(v_coords).all(axis=1)
    out["convex_hull_area_cm2"] = convex_hull_area(v_coords[finite]) if finite.sum() >= 3 else 0.0

    speed = compute_velocities(v_coords, v_time_s)
    path_length = float(np.nansum(np.linalg.norm(np.diff(v_coords, axis=0), axis=1)))
    out["path_length_cm"] = path_length
    out["mean_step_cm"] = path_length / max(len(v_coords) - 1, 1)
    out["speed_mean_cm_s"], out["speed_median_cm_s"], out["speed_p95_cm_s"] = summarize_list(speed)
    out["speed_max_cm_s"] = float(np.nanmax(speed))

    acc = np.diff(speed)
    acc_dt = np.diff(v_time_s)
    acc = np.concatenate(([np.nan], acc))
    acc_dt = np.concatenate(([np.nan], acc_dt))
    acc = acc / acc_dt
    out["acc_mean_cm_s2"], out["acc_median_cm_s2"], out["acc_p95_cm_s2"] = summarize_list(acc)

    out["drift_x_cm_per_s"] = lin_drift(v_time_s, v_coords[:, 0])
    out["drift_y_cm_per_s"] = lin_drift(v_time_s, v_coords[:, 1])

    gaps = compute_gaps(time_ms, valid_mask)
    out["missing_ratio"] = 1.0 - out["valid_ratio_strict"]
    out["gap_count"] = float(len(gaps))
    out["gap_longest_s"] = float(max(gaps)) if gaps else 0.0

    # Keep original (cm-based) saccade/fixation detection but with strict-valid mask already applied
    saccades = detect_saccades(v_coords, v_time_s, speed)
    out["saccade_count"] = float(len(saccades))
    out["saccade_rate_hz"] = float(len(saccades)) / max(out["duration_s"], 1e-3)
    if saccades:
        amps = [s["amplitude_cm"] for s in saccades]
        peaks = [s["peak_speed_cm_s"] for s in saccades]
        durs = [s["duration_s"] for s in saccades]
        out["saccade_amp_mean_cm"], out["saccade_amp_median_cm"], _ = summarize_list(amps)
        out["saccade_peak_mean_cm_s"], _, out["saccade_peak_p95_cm_s"] = summarize_list(peaks)
        out["saccade_duration_mean_s"], out["saccade_duration_median_s"], _ = summarize_list(durs)
    else:
        out["saccade_amp_mean_cm"] = float("nan")
        out["saccade_peak_mean_cm_s"] = float("nan")
        out["saccade_duration_mean_s"] = float("nan")
        out["saccade_amp_median_cm"] = float("nan")
        out["saccade_peak_p95_cm_s"] = float("nan")
        out["saccade_duration_median_s"] = float("nan")

    fixations = detect_fixations(v_coords, v_time_s, speed)
    out["fixation_count"] = float(len(fixations))
    out["fixation_rate_hz"] = float(len(fixations)) / max(out["duration_s"], 1e-3)
    if fixations:
        durs = [f["duration_s"] for f in fixations]
        disps = [f["dispersion_cm"] for f in fixations]
        out["fixation_mean_dur_s"], out["fixation_median_dur_s"], out["fixation_p95_dur_s"] = summarize_list(durs)
        out["fixation_mean_disp_cm"], out["fixation_median_disp_cm"], _ = summarize_list(disps)
        out["fixation_time_ratio"] = float(sum(durs)) / max(out["duration_s"], 1e-3)
    else:
        out["fixation_mean_dur_s"] = float("nan")
        out["fixation_median_dur_s"] = float("nan")
        out["fixation_p95_dur_s"] = float("nan")
        out["fixation_mean_disp_cm"] = float("nan")
        out["fixation_median_disp_cm"] = float("nan")
        out["fixation_time_ratio"] = 0.0

    # Normalize gaze metrics to a device-agnostic unit (fraction of screen diagonal).
    def norm(val: float) -> float:
        if norm_scale_cm is None or not np.isfinite(norm_scale_cm) or norm_scale_cm <= 0:
            return float("nan")
        return float(val) / norm_scale_cm if np.isfinite(val) else float("nan")

    def norm_area(val: float) -> float:
        if norm_scale_cm is None or not np.isfinite(norm_scale_cm) or norm_scale_cm <= 0:
            return float("nan")
        return float(val) / (norm_scale_cm ** 2) if np.isfinite(val) else float("nan")

    out.update(
        {
            "norm_scale_cm": norm_scale_cm if norm_scale_cm is not None else float("nan"),
            "norm_mean_x": norm(out.get("mean_x_cm", float("nan"))),
            "norm_mean_y": norm(out.get("mean_y_cm", float("nan"))),
            "norm_std_x": norm(out.get("std_x_cm", float("nan"))),
            "norm_std_y": norm(out.get("std_y_cm", float("nan"))),
            "norm_bbox_width": norm(out.get("bbox_width_cm", float("nan"))),
            "norm_bbox_height": norm(out.get("bbox_height_cm", float("nan"))),
            "norm_bbox_area": norm_area(out.get("bbox_area_cm2", float("nan"))),
            "norm_convex_hull_area": norm_area(out.get("convex_hull_area_cm2", float("nan"))),
            "norm_path_length": norm(out.get("path_length_cm", float("nan"))),
            "norm_speed_mean": norm(out.get("speed_mean_cm_s", float("nan"))),
            "norm_speed_median": norm(out.get("speed_median_cm_s", float("nan"))),
            "norm_speed_p95": norm(out.get("speed_p95_cm_s", float("nan"))),
            "norm_speed_max": norm(out.get("speed_max_cm_s", float("nan"))),
            "norm_acc_mean": norm(out.get("acc_mean_cm_s2", float("nan"))),
            "norm_acc_median": norm(out.get("acc_median_cm_s2", float("nan"))),
            "norm_acc_p95": norm(out.get("acc_p95_cm_s2", float("nan"))),
            "norm_saccade_amp_mean": norm(out.get("saccade_amp_mean_cm", float("nan"))),
            "norm_saccade_amp_median": norm(out.get("saccade_amp_median_cm", float("nan"))),
            "norm_saccade_peak_mean": norm(out.get("saccade_peak_mean_cm_s", float("nan"))),
            "norm_saccade_peak_p95": norm(out.get("saccade_peak_p95_cm_s", float("nan"))),
            "norm_fixation_mean_disp": norm(out.get("fixation_mean_disp_cm", float("nan"))),
            "norm_fixation_median_disp": norm(out.get("fixation_median_disp_cm", float("nan"))),
        }
    )

    return out
