#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Subject naming, version filtering, and Rawdata timestamp helpers for CoGaze."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


PRIVATE_SUBJECT_RE = re.compile(r"^(?P<device>.+)_(?P<subject>E\d+)_(?P<version>self|advanced)$", re.IGNORECASE)
RAW_SUBJECT_RE = re.compile(r"^(?P<device>.+)_(?P<subject>E\d+)$", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"^\d{8}T\d{6}$")


@dataclass
class SubjectParts:
    folder: str
    subject_code: str
    device_id: Optional[str]
    version: Optional[str]
    calibration_key: str


def parse_subject_folder(name: str) -> SubjectParts:
    match = PRIVATE_SUBJECT_RE.match(name)
    if match:
        device_id = match.group("device")
        subject_code = match.group("subject")
        version = match.group("version").lower()
        return SubjectParts(
            folder=name,
            subject_code=subject_code,
            device_id=device_id,
            version=version,
            calibration_key=f"{device_id}_{subject_code}",
        )
    raw_match = RAW_SUBJECT_RE.match(name)
    if raw_match:
        device_id = raw_match.group("device")
        subject_code = raw_match.group("subject")
        return SubjectParts(
            folder=name,
            subject_code=subject_code,
            device_id=device_id,
            version=None,
            calibration_key=f"{device_id}_{subject_code}",
        )
    return SubjectParts(
        folder=name,
        subject_code=name,
        device_id=None,
        version=None,
        calibration_key=name,
    )


def is_timestamp_name(value: str) -> bool:
    return bool(TIMESTAMP_RE.match(str(value).strip()))


def is_rawdata_root(data_root: Path) -> bool:
    if data_root.name.lower() == "rawdata":
        return True
    if not data_root.exists():
        return False
    return any(item.is_dir() and ((item / "self").is_dir() or (item / "advanced").is_dir()) for item in data_root.iterdir())


def rawdata_timestamps(subject_dir: Path, version: str) -> List[str]:
    phase_dir = subject_dir / version
    if not phase_dir.exists():
        return []
    return sorted(item.name for item in phase_dir.iterdir() if item.is_dir() and is_timestamp_name(item.name))


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%S")


def choose_rawdata_timestamp(subject_dir: Path, version: str, timestamp: Optional[str], assessment_phase: Optional[str]) -> Optional[str]:
    timestamps = rawdata_timestamps(subject_dir, version)
    if not timestamps:
        return None
    if not timestamp:
        return timestamps[-1]
    if timestamp in timestamps:
        return timestamp
    if assessment_phase == "advanced" and version == "self":
        target = parse_timestamp(timestamp)
        return min(timestamps, key=lambda item: abs(parse_timestamp(item) - target))
    return None


def subject_filter_matches(folder_name: str, subjects: Optional[List[str]]) -> bool:
    if not subjects:
        return True
    parts = parse_subject_folder(folder_name)
    candidates = {
        folder_name,
        parts.subject_code,
        parts.calibration_key,
    }
    if parts.device_id:
        candidates.add(parts.device_id)
    return bool(candidates.intersection(set(subjects)))


def version_filter_matches(folder_name: str, versions: Optional[List[str]]) -> bool:
    if not versions:
        return True
    parts = parse_subject_folder(folder_name)
    return (parts.version or "").lower() in {version.lower() for version in versions}
