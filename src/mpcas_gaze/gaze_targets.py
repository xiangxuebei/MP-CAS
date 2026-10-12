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


def align_event_targets(
    gaze_df: pd.DataFrame,
    dotinfo: List[dict],
    device_meta: Dict[str, object],
) -> Dict[str, float]:
    """
    Event-mode dotInfo (discrete jumps).
    Outputs task-relevant summaries: latency, gain, steady-state error.
    """
    if not dotinfo or gaze_df.empty:
        return {"mode": "events", "event_count": float(len(dotinfo))}

    g = gaze_df.copy()
    g["time_ms"] = pd.to_numeric(g["time_ms"], errors="coerce")
    g["x_cam"] = pd.to_numeric(g["x_cam"], errors="coerce")
    g["y_cam"] = pd.to_numeric(g["y_cam"], errors="coerce")
    ok = robust_valid_mask(g)
    g = g.loc[ok]
    if g.empty:
        return {"mode": "events", "event_count": float(len(dotinfo))}

    gaze_time = g["time_ms"].values.astype(float)
    hit_r = dynamic_hit_radius_cm(device_meta)

    # Extract dot times and auto-scale to ms
    starts = np.array([first_finite_value(d, ("half_second_time", "time_offset"), 0.0) for d in dotinfo], dtype=float)
    ends = np.array(
        [first_finite_value(d, ("disappear_time", "disappear_time_offset", "pulse_end_time")) for d in dotinfo],
        dtype=float,
    )
    starts, scale = _maybe_scale_time_to_ms(starts, gaze_time)
    ends, _ = _maybe_scale_time_to_ms(ends, gaze_time)
    durs = np.array([first_finite_value(d, ("duration",)) for d in dotinfo], dtype=float)
    durs, _ = _maybe_scale_time_to_ms(durs, gaze_time)
    missing_end = ~np.isfinite(ends)
    ends[missing_end] = starts[missing_end] + fill_nan(durs[missing_end], 1000.0)

    latencies: List[float] = []
    steady_err: List[float] = []
    gains: List[float] = []
    miss: List[int] = []

    prev_tx, prev_ty = None, None

    for i, event in enumerate(dotinfo):
        t0 = float(starts[i])
        t1 = float(ends[i]) if np.isfinite(ends[i]) else t0 + 1000.0
        tx = float(event.get("x_cam", np.nan))
        ty = float(event.get("y_cam", np.nan))
        if not (np.isfinite(tx) and np.isfinite(ty) and np.isfinite(t0)):
            miss.append(1)
            prev_tx, prev_ty = tx, ty
            continue

        # windows
        pre = g[(g["time_ms"] >= t0 - 500) & (g["time_ms"] < t0)]
        post = g[(g["time_ms"] >= t0) & (g["time_ms"] <= min(t0 + 1200, t1 + 200))]
        steady = g[(g["time_ms"] >= t0 + 400) & (g["time_ms"] <= min(t1, t0 + 2000))]

        # latency: first time within hit radius
        if not post.empty:
            dx = post["x_cam"].values - tx
            dy = post["y_cam"].values - ty
            dist = np.sqrt(dx * dx + dy * dy)
            within = dist <= hit_r
            if np.any(within):
                idx = int(np.argmax(within))
                latencies.append(float((post["time_ms"].values[idx] - t0) / 1000.0))
                miss.append(0)
            else:
                miss.append(1)
        else:
            miss.append(1)

        # steady-state error
        if not steady.empty:
            dx = steady["x_cam"].values - tx
            dy = steady["y_cam"].values - ty
            dist = np.sqrt(dx * dx + dy * dy)
            steady_err.append(float(np.nanmean(dist)))

        # gain (requires previous target)
        if prev_tx is not None and np.isfinite(prev_tx) and np.isfinite(prev_ty) and (not pre.empty) and (not steady.empty):
            step = float(math.hypot(tx - prev_tx, ty - prev_ty))
            if step > 1e-6:
                pre_center = np.nanmedian(pre[["x_cam", "y_cam"]].values, axis=0)
                post_center = np.nanmedian(steady[["x_cam", "y_cam"]].values, axis=0)
                obs = float(np.linalg.norm(post_center - pre_center))
                gains.append(obs / step)

        prev_tx, prev_ty = tx, ty

    out = {
        "mode": "events",
        "event_count": float(len(dotinfo)),
        "hit_radius_cm": float(hit_r),
        "miss_ratio": float(np.mean(miss)) if miss else float("nan"),
        "latency_median_s": float(np.nanmedian(latencies)) if latencies else float("nan"),
        "latency_mean_s": float(np.nanmean(latencies)) if latencies else float("nan"),
        "steady_error_mean_cm": float(np.nanmean(steady_err)) if steady_err else float("nan"),
        "steady_error_median_cm": float(np.nanmedian(steady_err)) if steady_err else float("nan"),
        "gain_mean": float(np.nanmean(gains)) if gains else float("nan"),
        "gain_median": float(np.nanmedian(gains)) if gains else float("nan"),
    }
    return out


def align_timeline_targets(
    gaze_df: pd.DataFrame,
    dotinfo: List[dict],
    device_meta: Dict[str, object],
) -> Dict[str, float]:
    """
    Timeline-mode dotInfo (continuous target trajectory).
    Computes pursuit-like features: error, gain, lag.
    """
    dot_df = pd.DataFrame(dotinfo)
    if dot_df.empty or "time_offset" not in dot_df:
        return {"mode": "timeline", "aligned_samples": 0.0}

    g = gaze_df.copy()
    g["time_ms"] = pd.to_numeric(g["time_ms"], errors="coerce")
    g["x_cam"] = pd.to_numeric(g["x_cam"], errors="coerce")
    g["y_cam"] = pd.to_numeric(g["y_cam"], errors="coerce")
    ok = robust_valid_mask(g)
    g = g.loc[ok]
    if g.empty:
        return {"mode": "timeline", "aligned_samples": 0.0}

    gaze_time = g["time_ms"].values.astype(float)

    dot_df = dot_df.rename(columns={"time_offset": "time_ms"})
    dot_df["time_ms"] = pd.to_numeric(dot_df["time_ms"], errors="coerce")
    dot_df["x_cam"] = pd.to_numeric(dot_df.get("x_cam"), errors="coerce")
    dot_df["y_cam"] = pd.to_numeric(dot_df.get("y_cam"), errors="coerce")
    dot_df = dot_df.dropna(subset=["time_ms", "x_cam", "y_cam"])
    if dot_df.empty:
        return {"mode": "timeline", "aligned_samples": 0.0}

    times = dot_df["time_ms"].values.astype(float)
    times, scale = _maybe_scale_time_to_ms(times, gaze_time)
    dot_df = dot_df.copy()
    dot_df["time_ms"] = times

    # Merge-asof to align
    dt_g = np.nanmedian(np.diff(np.sort(gaze_time)))
    tol = int(round(max(80.0, 0.8 * dt_g))) if np.isfinite(dt_g) else 120
    merged = pd.merge_asof(
        g.sort_values("time_ms"),
        dot_df[["time_ms", "x_cam", "y_cam"]].sort_values("time_ms"),
        on="time_ms",
        direction="nearest",
        tolerance=tol,
        suffixes=("_g", "_t"),
    )
    merged = merged.dropna(subset=["x_cam_g", "y_cam_g", "x_cam_t", "y_cam_t"])
    if merged.empty:
        return {"mode": "timeline", "aligned_samples": 0.0}

    dx = merged["x_cam_g"].values - merged["x_cam_t"].values
    dy = merged["y_cam_g"].values - merged["y_cam_t"].values
    dist = np.sqrt(dx * dx + dy * dy)

    # Gain/lag (guard against sparse alignment or nearly-static targets)
    gain = float("nan")
    corr0 = float("nan")
    lag_ms = float("nan")
    n_aligned = len(merged)
    # Only compute pursuit-like metrics if we have enough aligned samples
    if n_aligned >= 30:
        t_ms = merged["time_ms"].values.astype(float)
        fs = estimate_sample_rate(t_ms)
        # If the aligned sampling rate collapses (e.g., due to bad time units), skip gain/lag
        if np.isfinite(fs) and fs >= 5.0:
            t_s = t_ms / 1000.0
            coords_g = merged[["x_cam_g", "y_cam_g"]].values.astype(float)
            coords_t = merged[["x_cam_t", "y_cam_t"]].values.astype(float)

            vg = compute_velocities(coords_g - np.nanmedian(coords_g, axis=0), t_s)
            vt = compute_velocities(coords_t - np.nanmedian(coords_t, axis=0), t_s)
            vt_med = float(np.nanmedian(vt))
            vg_med = float(np.nanmedian(vg))
            # Velocity gain should be close to ~1; treat extreme ratios as invalid
            if np.isfinite(vt_med) and vt_med > 1e-3 and np.isfinite(vg_med):
                tmp_gain = vg_med / vt_med
                gain = float(tmp_gain) if (0.0 <= tmp_gain <= 5.0) else float("nan")

            # Lag via cross-correlation of x (robust enough at ~10Hz)
            gx = coords_g[:, 0] - np.nanmedian(coords_g[:, 0])
            tx = coords_t[:, 0] - np.nanmedian(coords_t[:, 0])
            mask = np.isfinite(gx) & np.isfinite(tx)
            if mask.sum() >= 30:
                gx0 = gx[mask] - np.mean(gx[mask])
                tx0 = tx[mask] - np.mean(tx[mask])
                # search lags within +-800ms and require a meaningful correlation peak
                max_lag = int(round(min((0.8 * fs), len(gx0) // 4)))
                if max_lag >= 1:
                    best_r = -1e9
                    best_l = 0
                    for l in range(-max_lag, max_lag + 1):
                        if l < 0:
                            a = gx0[-l:]
                            b = tx0[: len(a)]
                        elif l > 0:
                            a = gx0[:-l]
                            b = tx0[l:]
                        else:
                            a = gx0
                            b = tx0
                        if len(a) < 20:
                            continue
                        r = float(np.corrcoef(a, b)[0, 1])
                        if np.isfinite(r) and r > best_r:
                            best_r = r
                            best_l = l
                    if np.isfinite(best_r) and best_r >= 0.30:
                        corr0 = float(best_r)
                        lag_ms = float(1000.0 * best_l / fs)
    hit_r = dynamic_hit_radius_cm(device_meta)
    hit_ratio = float(np.mean(dist <= hit_r)) if dist.size else float("nan")
    rmse = float(np.sqrt(np.nanmean(dist * dist))) if dist.size else float("nan")

    out = {
        "mode": "timeline",
        "aligned_samples": float(len(merged)),
        "tolerance_ms": float(tol),
        "hit_radius_cm": float(hit_r),
        "mean_error_cm": float(np.nanmean(dist)),
        "median_error_cm": float(np.nanmedian(dist)),
        "p95_error_cm": float(np.nanpercentile(dist, 95)),
        "rmse_error_cm": rmse,
        "hit_ratio": hit_ratio,
        "vel_gain": gain,
        "x_corr0": corr0,
        "x_lag_ms": lag_ms,
    }
    return out


def compute_target_tracking_metrics(
    gaze_df: pd.DataFrame,
    dotinfo_path: Optional[Path],
    device_meta: Dict[str, object],
) -> Dict[str, float]:
    if dotinfo_path is None or not dotinfo_path.exists():
        return {"mode": "none"}
    dotinfo = load_json(dotinfo_path)
    dtype = classify_dotinfo(dotinfo)
    if dtype == "events":
        return align_event_targets(gaze_df, dotinfo, device_meta)
    if dtype == "timeline":
        return align_timeline_targets(gaze_df, dotinfo, device_meta)
    return {"mode": "none"}

