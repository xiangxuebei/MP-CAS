#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rawdata phase discovery for multimodal feature extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from mpcas_tasks.rawdata import available_rawdata_timestamps, parse_analysis_timestamp
except ImportError:  # pragma: no cover - supports direct script execution
    from rawdata import available_rawdata_timestamps, parse_analysis_timestamp


PRIVATE_SUBJECT_SUFFIXES = {"self", "advanced"}


@dataclass
class PhaseRecord:
    folder_subject: str
    subject: str
    subject_code: str
    device_id: str
    version: str
    timestamp: str
    task: str
    phase: str
    phase_dir: Path


def parse_private_subject(folder_name: str) -> Tuple[str, str, str, str]:
    parts = folder_name.split("_")
    if len(parts) >= 3 and parts[-1].lower() in PRIVATE_SUBJECT_SUFFIXES and parts[-2].upper().startswith("E"):
        version = parts[-1].lower()
        subject_code = parts[-2].upper()
        device_id = "_".join(parts[:-2])
        subject = f"{device_id}_{subject_code}"
        return subject, subject_code, device_id, version
    if len(parts) >= 2 and parts[-1].upper().startswith("E"):
        subject_code = parts[-1].upper()
        device_id = "_".join(parts[:-1])
        subject = f"{device_id}_{subject_code}" if device_id else subject_code
        return subject, subject_code, device_id, ""
    match = re.search(r"E\d+", folder_name, flags=re.IGNORECASE)
    subject_code = match.group(0).upper() if match else folder_name
    return folder_name, subject_code, "", ""


def choose_rawdata_timestamp(
    subject_dir: Path,
    version: str,
    timestamp: Optional[str],
    assessment_phase: Optional[str],
) -> Optional[str]:
    timestamps = available_rawdata_timestamps(subject_dir, version)
    if not timestamps:
        return None
    if not timestamp:
        return timestamps[-1]
    if timestamp in timestamps:
        return timestamp
    if assessment_phase == "advanced" and version == "self":
        target = parse_analysis_timestamp(timestamp)
        return min(timestamps, key=lambda item: abs(parse_analysis_timestamp(item) - target))
    return None


def discover_phase_records(
    data_root: Path,
    subjects: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    phases: Optional[List[str]] = None,
    include_t001: bool = True,
    versions: Optional[List[str]] = None,
    timestamp: Optional[str] = None,
    assessment_phase: Optional[str] = None,
) -> List[PhaseRecord]:
    """Discover task phase directories from the Rawdata layout only."""
    subject_filter = {s.strip() for s in subjects or [] if s.strip()}
    task_filter = set(tasks or [])
    phase_filter = set(phases or [])
    version_filter = {v.strip().lower() for v in versions or [] if v.strip()}
    records: List[PhaseRecord] = []
    if not data_root.exists():
        return records

    for subject_dir in sorted(data_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject, subject_code, device_id, _ = parse_private_subject(subject_dir.name)
        candidates = {subject_dir.name, subject, subject_code, device_id}
        if subject_filter and not candidates.intersection(subject_filter):
            continue
        candidate_versions = ["self", "advanced"]
        if version_filter:
            candidate_versions = [version for version in candidate_versions if version in version_filter]
        for version in candidate_versions:
            selected_timestamp = choose_rawdata_timestamp(subject_dir, version, timestamp, assessment_phase)
            if not selected_timestamp:
                continue
            timestamp_dir = subject_dir / version / selected_timestamp
            folder_subject = f"{subject}_{version}"
            for task_dir in sorted(timestamp_dir.iterdir()):
                if not task_dir.is_dir() or not task_dir.name.startswith("T"):
                    continue
                if task_dir.name == "T001" and not include_t001:
                    continue
                if task_filter and task_dir.name not in task_filter:
                    continue
                for phase_dir in sorted(task_dir.iterdir()):
                    if not phase_dir.is_dir():
                        continue
                    if phase_filter and phase_dir.name not in phase_filter:
                        continue
                    records.append(
                        PhaseRecord(
                            folder_subject=folder_subject,
                            subject=subject,
                            subject_code=subject_code,
                            device_id=device_id,
                            version=version,
                            timestamp=selected_timestamp,
                            task=task_dir.name,
                            phase=phase_dir.name,
                            phase_dir=phase_dir,
                        )
                    )
    return records
