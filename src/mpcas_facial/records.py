"""Find task videos without embedding study-specific absolute paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoRecord:
    subject: str
    task: str
    phase: str
    path: Path
    record_key: str


def discover_videos(root: Path, tasks: set[str] | None = None) -> list[VideoRecord]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    records: list[VideoRecord] = []
    for path in sorted(root.rglob("video.mp4")):
        phase_dir = path.parent
        task_dir = phase_dir.parent
        if not re.fullmatch(r"T\d{3}", task_dir.name):
            continue
        if tasks and task_dir.name not in tasks:
            continue
        if phase_dir.name.lower() in {"formalitem", "practiceitem"}:
            continue
        names = task_dir.relative_to(root).parts[:-1]
        subject = next((name.upper() for name in names if re.fullmatch(r"[A-Za-z]{1,3}\d{4,}", name)), None)
        if subject is None:
            subject = next((match.group(0).upper() for name in names if (match := re.search(r"E\d{4,}", name, re.I))), None)
        if subject is None:
            continue
        records.append(VideoRecord(subject, task_dir.name, phase_dir.name, path, path.relative_to(root).as_posix()))
    return records
