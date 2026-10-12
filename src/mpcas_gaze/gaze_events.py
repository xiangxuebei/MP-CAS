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


def compute_gaps(time_ms: np.ndarray, valid_mask: np.ndarray) -> List[float]:
    gaps = []
    start_idx: Optional[int] = None
    for idx, ok in enumerate(valid_mask):
        if not ok and start_idx is None:
            start_idx = idx
        elif ok and start_idx is not None:
            gaps.append((time_ms[idx] - time_ms[start_idx]) / 1000.0)
            start_idx = None
    if start_idx is not None and len(time_ms):
        gaps.append((time_ms[-1] - time_ms[start_idx]) / 1000.0)
    return gaps


def compute_velocities(coords: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    if len(coords) < 2:
        return np.full(len(coords), np.nan)
    dt = np.diff(time_s)
    dt = np.concatenate(([np.nan], dt))
    dt[dt == 0] = np.nan
    dxy = np.diff(coords, axis=0)
    dxy = np.vstack((np.full((1, coords.shape[1]), np.nan), dxy))
    speed = np.linalg.norm(dxy, axis=1) / dt
    speed[0] = np.nan
    return speed


def summarize_list(values: Iterable[float]) -> Tuple[float, float, float]:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return float(np.nanmean(arr)), float(np.nanmedian(arr)), float(np.nanpercentile(arr, 95))


def robust_valid_mask(df: pd.DataFrame) -> np.ndarray:
    return (
        (df.get("valid", 0).astype(int).values == 1)
        & np.isfinite(df.get("x_cam").values)
        & np.isfinite(df.get("y_cam").values)
        & np.isfinite(df.get("time_ms").values)
    )


def compute_blink_stats(time_ms: np.ndarray, ok_mask: np.ndarray) -> Dict[str, float]:
    """
    Blink/tracking-loss stats from invalid segments.
    At ~10Hz, the minimum resolvable gap is ~0.1s; we treat
    0.05-0.8s as "blink-like", longer as "tracking loss".
    """
    gaps = compute_gaps(time_ms, ok_mask)
    if not gaps:
        return {
            "blink_count": 0.0,
            "blink_rate_hz": 0.0,
            "blink_mean_dur_s": 0.0,
            "blink_p95_dur_s": 0.0,
            "tracking_loss_total_s": 0.0,
            "tracking_loss_count": 0.0,
        }

    gaps_arr = np.asarray(gaps, dtype=float)
    blink = gaps_arr[(gaps_arr >= 0.05) & (gaps_arr <= 0.8)]
    loss = gaps_arr[gaps_arr > 0.8]

    dur_s = (time_ms[-1] - time_ms[0]) / 1000.0 if len(time_ms) > 1 else 0.0
    out = {
        "blink_count": float(blink.size),
        "blink_rate_hz": float(blink.size) / max(dur_s, 1e-3),
        "blink_mean_dur_s": float(np.mean(blink)) if blink.size else 0.0,
        "blink_p95_dur_s": float(np.percentile(blink, 95)) if blink.size else 0.0,
        "tracking_loss_total_s": float(np.sum(loss)) if loss.size else 0.0,
        "tracking_loss_count": float(loss.size),
    }
    return out


def spatial_entropy(coords_norm: np.ndarray, grid: int = 10) -> Tuple[float, float]:
    """
    coords_norm: Nx2 in [-0.5, 0.5] approximately (diag-normalized and centered).
    Returns: (occupancy_entropy_normed, transition_entropy_normed)
    """
    if len(coords_norm) < 5:
        return float("nan"), float("nan")
    x = coords_norm[:, 0]
    y = coords_norm[:, 1]
    # Clip to a reasonable range to reduce outlier impact
    x = np.clip(x, -0.75, 0.75)
    y = np.clip(y, -0.75, 0.75)

    # Bin to grid
    bins = np.linspace(-0.75, 0.75, grid + 1)
    xi = np.digitize(x, bins) - 1
    yi = np.digitize(y, bins) - 1
    xi = np.clip(xi, 0, grid - 1)
    yi = np.clip(yi, 0, grid - 1)
    cell = yi * grid + xi

    # Occupancy entropy
    counts = np.bincount(cell, minlength=grid * grid).astype(float)
    p = counts / max(counts.sum(), 1.0)
    p = p[p > 0]
    occ_ent = -float(np.sum(p * np.log(p)))
    occ_ent_norm = occ_ent / math.log(grid * grid)

    # Transition entropy (first-order)
    trans = cell[1:] * (grid * grid) + cell[:-1]
    tcounts = np.bincount(trans, minlength=(grid * grid) ** 2).astype(float)
    tp = tcounts / max(tcounts.sum(), 1.0)
    tp = tp[tp > 0]
    tr_ent = -float(np.sum(tp * np.log(tp)))
    tr_ent_norm = tr_ent / math.log((grid * grid) ** 2)

    return occ_ent_norm, tr_ent_norm


def detect_ivt_events(
    coords_deg: np.ndarray,
    time_s: np.ndarray,
    vel_deg_s: np.ndarray,
    sacc_thr: float = 30.0,
    fix_thr: float = 10.0,
    min_fix_dur: float = 0.10,
    min_sacc_dur: float = 0.02,
    merge_gap: float = 0.04,
) -> Tuple[List[dict], List[dict]]:
    """
    I-VT event detection in deg/s.
    Returns (saccades, fixations).
    """
    n = len(vel_deg_s)
    if n == 0:
        return [], []

    # Identify saccade samples
    v = np.array(vel_deg_s, copy=True)
    v[~np.isfinite(v)] = -np.inf
    is_sacc = v >= sacc_thr
    is_fix = (v >= 0) & (v <= fix_thr)

    def segments(mask: np.ndarray) -> List[Tuple[int, int]]:
        segs: List[Tuple[int, int]] = []
        i = 0
        while i < n:
            if mask[i]:
                s = i
                while i + 1 < n and mask[i + 1]:
                    i += 1
                e = i
                segs.append((s, e))
            i += 1
        return segs

    sacc_segs = segments(is_sacc)
    fix_segs = segments(is_fix)

    # Merge segments separated by very small gaps (helps low fps)
    def merge(segs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not segs:
            return []
        merged = [segs[0]]
        for s, e in segs[1:]:
            ps, pe = merged[-1]
            gap = time_s[s] - time_s[pe]
            if np.isfinite(gap) and gap <= merge_gap:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))
        return merged

    sacc_segs = merge(sacc_segs)
    fix_segs = merge(fix_segs)

    saccades: List[dict] = []
    for s, e in sacc_segs:
        dur = float(time_s[e] - time_s[s])
        if dur >= min_sacc_dur:
            amp = float(np.linalg.norm(coords_deg[e] - coords_deg[s]))
            peak_v = float(np.nanmax(vel_deg_s[s : e + 1]))
            saccades.append(
                {
                    "start_idx": s,
                    "end_idx": e,
                    "duration_s": dur,
                    "amplitude_deg": amp,
                    "peak_speed_deg_s": peak_v,
                }
            )

    fixations: List[dict] = []
    for s, e in fix_segs:
        dur = float(time_s[e] - time_s[s])
        if dur >= min_fix_dur:
            seg = coords_deg[s : e + 1]
            disp = float(np.linalg.norm(np.nanmax(seg, axis=0) - np.nanmin(seg, axis=0)))
            fixations.append(
                {
                    "start_idx": s,
                    "end_idx": e,
                    "duration_s": dur,
                    "dispersion_deg": disp,
                    "samples": int(e - s + 1),
                }
            )
    return saccades, fixations


def detect_saccades(
    coords: np.ndarray,
    time_s: np.ndarray,
    speed: np.ndarray,
    min_dur: float = 0.02,
    lambda_thr: float = 5.0,
) -> List[dict]:
    saccades: List[dict] = []
    if len(speed) == 0:
        return saccades

    med = np.nanmedian(speed)
    mad = np.nanmedian(np.abs(speed - med))
    thr = med + lambda_thr * mad
    thr = max(thr, 50.0)  # minimum speed threshold

    above = np.nan_to_num(speed) > thr
    idx = 0
    while idx < len(speed):
        if above[idx]:
            start = idx
            while idx + 1 < len(speed) and above[idx + 1]:
                idx += 1
            end = idx
            dur = time_s[end] - time_s[start]
            if dur >= min_dur:
                amp = float(np.linalg.norm(coords[end] - coords[start]))
                peak_v = float(np.nanmax(speed[start : end + 1]))
                saccades.append(
                    {
                        "start_idx": start,
                        "end_idx": end,
                        "duration_s": float(dur),
                        "amplitude_cm": amp,
                        "peak_speed_cm_s": peak_v,
                    }
                )
        idx += 1
    return saccades


def detect_fixations(
    coords: np.ndarray,
    time_s: np.ndarray,
    speed: np.ndarray,
    min_dur: float = 0.12,
) -> List[dict]:
    fixations: List[dict] = []
    if len(speed) == 0:
        return fixations

    med = np.nanmedian(speed)
    mad = np.nanmedian(np.abs(speed - med))
    thr = med + 2.0 * mad
    thr = max(thr, 15.0)

    speed_clean = np.array(speed, copy=True)
    speed_clean[np.isnan(speed_clean)] = np.inf
    below = speed_clean < thr
    start: Optional[int] = None
    for idx, flag in enumerate(below):
        if flag and start is None:
            start = idx
        elif (not flag or idx == len(below) - 1) and start is not None:
            end = idx if flag else idx - 1
            dur = time_s[end] - time_s[start]
            if dur >= min_dur:
                seg = coords[start : end + 1]
                disp = float(np.linalg.norm(np.nanmax(seg, axis=0) - np.nanmin(seg, axis=0)))
                fixations.append(
                    {
                        "start_idx": start,
                        "end_idx": end,
                        "duration_s": float(dur),
                        "dispersion_cm": disp,
                        "samples": int(end - start + 1),
                    }
                )
            start = None
    return fixations
