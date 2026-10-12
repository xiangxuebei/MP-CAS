#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rawdata path discovery and upload helpers.

This module owns the server-side Rawdata contract:

Rawdata/{subject_id}/{self|advanced}/{YYYYMMDDTHHMMSS}/Txxx/{phase}/...
"""

from __future__ import annotations

import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_PHASES = {"self", "advanced"}
TIMESTAMP_PATTERN = re.compile(r"^\d{8}T\d{6}$")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def parse_analysis_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%S")


def is_timestamp_name(value: str) -> bool:
    return bool(TIMESTAMP_PATTERN.match(str(value).strip()))


def rawdata_subject_candidates(raw_root: Path, subject_id: str) -> List[Path]:
    if not raw_root.exists():
        return []
    exact = raw_root / subject_id
    if exact.is_dir():
        return [exact]
    subject_upper = subject_id.upper()
    matches = [
        item
        for item in raw_root.iterdir()
        if item.is_dir()
        and (
            item.name.upper() == subject_upper
            or item.name.upper().endswith("_" + subject_upper)
        )
    ]
    return sorted(matches, key=lambda item: item.name)


def resolve_rawdata_subject_dir(raw_root: Path, subject_id: str) -> Path:
    candidates = rawdata_subject_candidates(raw_root, subject_id)
    if not candidates:
        raise FileNotFoundError(f"rawdata subject folder not found: {raw_root / subject_id}")
    if len(candidates) > 1:
        names = ", ".join(item.name for item in candidates)
        raise ValueError(f"multiple rawdata subject folders match {subject_id}: {names}")
    return candidates[0]


def available_rawdata_timestamps(subject_dir: Path, assessment_phase: str) -> List[str]:
    phase_dir = subject_dir / assessment_phase
    if not phase_dir.exists():
        return []
    return sorted(item.name for item in phase_dir.iterdir() if item.is_dir() and is_timestamp_name(item.name))


def latest_rawdata_timestamp(subject_dir: Path, assessment_phase: str) -> str:
    timestamps = available_rawdata_timestamps(subject_dir, assessment_phase)
    if not timestamps:
        raise FileNotFoundError(f"no rawdata timestamp folders under: {subject_dir / assessment_phase}")
    return timestamps[-1]


def closest_rawdata_timestamp(subject_dir: Path, assessment_phase: str, target_timestamp: str) -> str:
    timestamps = available_rawdata_timestamps(subject_dir, assessment_phase)
    if not timestamps:
        raise FileNotFoundError(f"no rawdata timestamp folders under: {subject_dir / assessment_phase}")
    target = parse_analysis_timestamp(target_timestamp)
    return min(timestamps, key=lambda item: abs(parse_analysis_timestamp(item) - target))


def is_rawdata_root(data_root: Path, subject_id: Optional[str] = None) -> bool:
    if data_root.name.lower() == "rawdata":
        return True
    if subject_id:
        return bool(rawdata_subject_candidates(data_root, subject_id))
    if not data_root.exists():
        return False
    return any(item.is_dir() and ((item / "self").is_dir() or (item / "advanced").is_dir()) for item in data_root.iterdir())


def list_rawdata_tasks(timestamp_dir: Path) -> List[str]:
    if not timestamp_dir.exists():
        raise FileNotFoundError(f"rawdata timestamp folder not found: {timestamp_dir}")
    return sorted(item.name for item in timestamp_dir.iterdir() if item.is_dir() and item.name.startswith("T"))


def select_rawdata_sources(raw_root: Path, subject_id: str, assessment_phase: str, timestamp: str) -> Dict[str, Any]:
    subject_dir = resolve_rawdata_subject_dir(raw_root, subject_id)
    canonical_subject_id = subject_dir.name
    selected: Dict[str, Any] = {
        "rawdata_root": str(raw_root),
        "rawdata_subject_dir": str(subject_dir),
        "subject_id": canonical_subject_id,
        "assessment_phase": assessment_phase,
        "timestamp": timestamp,
        "sources": {},
    }

    if assessment_phase == "self":
        source = subject_dir / "self" / timestamp
        selected["sources"]["self"] = {
            "timestamp": timestamp,
            "path": str(source),
            "tasks": list_rawdata_tasks(source),
        }
        return selected

    advanced_source = subject_dir / "advanced" / timestamp
    selected["sources"]["advanced"] = {
        "timestamp": timestamp,
        "path": str(advanced_source),
        "tasks": list_rawdata_tasks(advanced_source),
    }

    self_timestamp = closest_rawdata_timestamp(subject_dir, "self", timestamp)
    self_source = subject_dir / "self" / self_timestamp
    selected["sources"]["self"] = {
        "timestamp": self_timestamp,
        "path": str(self_source),
        "tasks": list_rawdata_tasks(self_source),
        "selection": "closest_to_advanced_timestamp",
    }
    return selected


def subject_folder_matches(folder_name: str, subject_id: str) -> bool:
    folder_upper = folder_name.upper()
    subject_upper = subject_id.upper()
    return folder_upper == subject_upper or folder_upper.endswith("_" + subject_upper)


def safe_zip_parts(filename: str) -> List[str]:
    normalized = filename.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"invalid absolute zip path: {filename}")
    parts = [part for part in normalized.split("/") if part and part != "."]
    if any(part == ".." or ":" in part for part in parts):
        raise ValueError(f"unsafe zip path: {filename}")
    return parts


def is_within_directory(child: Path, parent: Path) -> bool:
    child_resolved = child.resolve()
    parent_resolved = parent.resolve()
    return child_resolved == parent_resolved or parent_resolved in child_resolved.parents


def rawdata_relative_parts(parts: List[str], subject_id: str) -> List[str]:
    if parts and parts[0].lower() == "rawdata":
        parts = parts[1:]
    subject_index = next((idx for idx, item in enumerate(parts) if subject_folder_matches(item, subject_id)), None)
    if subject_index is not None:
        return parts[subject_index:]
    if len(parts) >= 2 and parts[0].lower() in VALID_PHASES and is_timestamp_name(parts[1]):
        return [subject_id, *parts]
    raise ValueError(
        "zip must contain Rawdata-style paths like "
        f"{subject_id}/self/20260427T153045/T004/formal/video.mp4"
    )


def extract_rawdata_zip(zip_path: Path, raw_root: Path, subject_id: str, overwrite: bool = False) -> Dict[str, Any]:
    ensure_dir(raw_root)
    staging_root = raw_root / "_upload_staging" / f"{subject_id}_{now_timestamp()}"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    ensure_dir(staging_root)
    extracted_timestamps = set()
    subject_folders = set()
    file_count = 0

    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = safe_zip_parts(info.filename)
                if not parts or parts[0] == "__MACOSX":
                    continue
                relative_parts = rawdata_relative_parts(parts, subject_id)
                if len(relative_parts) < 4:
                    continue
                raw_subject, phase, timestamp = relative_parts[:3]
                phase = phase.lower()
                if not subject_folder_matches(raw_subject, subject_id):
                    raise ValueError(f"zip contains folder for another subject: {raw_subject}")
                if phase not in VALID_PHASES:
                    raise ValueError(f"invalid rawdata assessment phase in zip: {phase}")
                if not is_timestamp_name(timestamp):
                    raise ValueError(f"invalid rawdata timestamp in zip: {timestamp}")

                normalized_parts = [raw_subject, phase, timestamp, *relative_parts[3:]]
                target = staging_root.joinpath(*normalized_parts)
                if not is_within_directory(target, staging_root):
                    raise ValueError(f"unsafe zip extraction path: {info.filename}")
                ensure_dir(target.parent)
                with zf.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                subject_folders.add(raw_subject)
                extracted_timestamps.add((raw_subject, phase, timestamp))
                file_count += 1

        if not extracted_timestamps:
            raise ValueError("uploaded zip contains no valid Rawdata files")

        moved_dirs: List[str] = []
        for raw_subject, phase, timestamp in sorted(extracted_timestamps):
            source_dir = staging_root / raw_subject / phase / timestamp
            target_dir = raw_root / raw_subject / phase / timestamp
            if target_dir.exists():
                if not overwrite:
                    raise FileExistsError(f"rawdata timestamp folder already exists: {target_dir}")
                if not is_within_directory(target_dir, raw_root):
                    raise ValueError(f"refusing to overwrite outside rawdata root: {target_dir}")
                shutil.rmtree(target_dir)
            ensure_dir(target_dir.parent)
            shutil.move(str(source_dir), str(target_dir))
            moved_dirs.append(str(target_dir))

        return {
            "subject_id": subject_id,
            "rawdata_root": str(raw_root),
            "subject_folders": sorted(subject_folders),
            "extracted_timestamps": [
                {"subject": item[0], "assessment_phase": item[1], "timestamp": item[2]}
                for item in sorted(extracted_timestamps)
            ],
            "extracted_folders": moved_dirs,
            "file_count": file_count,
            "overwrite": overwrite,
        }
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
