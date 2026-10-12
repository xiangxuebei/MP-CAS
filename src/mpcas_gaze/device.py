#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Device metadata defaults and parsing for gaze feature extraction."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


PHONE_DEFAULT_DEVICE = {
    "device_model": "SM-S9380",
    "screen_width_cm": 7.3,
    "screen_height_cm": 15.799999,
    "screen_diag_cm": 17.40488346,
    "screen_width_px": 1080.0,
    "screen_height_px": 2340.0,
    "device_class": "phone",
}

TABLET_DEFAULT_DEVICE = {
    "device_model": None,
    "screen_width_cm": 31.394033,
    "screen_height_cm": 19.603825,
    "screen_diag_cm": 37.01209617,
    "screen_width_px": 2960.0,
    "screen_height_px": 1848.0,
    "device_class": "tablet",
}

VIEWING_DISTANCE_CM = {"phone": 40.0, "tablet": 50.0, "unknown": 45.0}


def parse_device_info(info: Optional[dict]) -> Dict[str, object]:
    def to_float(val) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    device = (info or {}).get("device_info", {}) or {}
    screen_cm = device.get("screen_size_cm", {}) or {}
    screen_px = device.get("screen_resolution", {}) or {}

    w_cm = to_float(screen_cm.get("width"))
    h_cm = to_float(screen_cm.get("height"))
    diag_cm = math.hypot(w_cm, h_cm) if np.isfinite(w_cm) and np.isfinite(h_cm) else float("nan")

    device_class = "unknown"
    if np.isfinite(diag_cm):
        device_class = "phone" if diag_cm < 20 else "tablet"

    return {
        "device_model": device.get("model"),
        "screen_width_cm": w_cm,
        "screen_height_cm": h_cm,
        "screen_diag_cm": diag_cm,
        "screen_width_px": to_float(screen_px.get("width")),
        "screen_height_px": to_float(screen_px.get("height")),
        "device_class": device_class,
    }


def default_device_for_subject(subject: str) -> Optional[Dict[str, object]]:
    code = subject.upper()
    if code.startswith(("E00", "E01", "E02", "E03")):
        return dict(PHONE_DEFAULT_DEVICE)
    if code.startswith(("E04", "E05", "E06")):
        return dict(TABLET_DEFAULT_DEVICE)
    return None
