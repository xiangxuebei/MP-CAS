#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audio extraction orchestration for the multimodal pipeline."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from mpcas_tasks.processes import run_command
    from mpcas_tasks.records import discover_phase_records
    from mpcas_tasks.stage_logging import PIPELINE_VERBOSE, StageLogger, format_seconds, optional_stage
except ImportError:  # pragma: no cover - supports direct script execution
    from processes import run_command
    from records import discover_phase_records
    from stage_logging import PIPELINE_VERBOSE, StageLogger, format_seconds, optional_stage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_FEATURE_FILENAME = "audio_features.csv"
TASK_NAME_RE = re.compile(r"^T\d{3}$")
PRIVATE_SUBJECT_SUFFIXES = {"self", "advanced"}
EXCLUDED_PHASES = {"formalitem", "practiceitem"}
AUDIO_PREPROCESS_VERSION = "task-leading-trim-v1"
AUDIO_LEADING_TRIM_SECONDS = {
    ("T004", "formal"): 46.0,
    ("T004", "practice"): 17.0,
    ("T009", "formal"): 15.0,
    ("T013", "formal"): 24.0,
    ("T013", "practice"): 18.0,
}


@dataclass
class AudioPhaseRecord:
    subject: str
    folder_subject: str
    subject_code: str
    device_id: str
    version: str
    timestamp: str
    task: str
    phase: str
    phase_dir: Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ffmpeg_executable() -> str:
    configured = os.environ.get("FFMPEG_EXE", "").strip()
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        imageio_binary = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if imageio_binary.is_file():
            return str(imageio_binary)
    except (ImportError, RuntimeError):
        pass
    raise FileNotFoundError("ffmpeg not found. Install ffmpeg or add it to PATH.")


def parse_samsung_session(folder_name: str, subject_bucket: str) -> Dict[str, str]:
    parts = folder_name.split("_")
    version = ""
    if len(parts) >= 3 and parts[-1].lower() in PRIVATE_SUBJECT_SUFFIXES and parts[-2].upper().startswith("E"):
        version = parts[-1].lower()
        subject_code = parts[-2].upper()
        device_id = "_".join(parts[:-2])
    elif len(parts) >= 2 and parts[-1].upper().startswith("E"):
        subject_code = parts[-1].upper()
        device_id = "_".join(parts[:-1])
    else:
        match = re.search(r"E\d+", folder_name, flags=re.IGNORECASE)
        subject_code = match.group(0).upper() if match else subject_bucket
        device_id = ""
    return {
        "subject": subject_bucket,
        "folder_subject": folder_name,
        "subject_code": subject_code,
        "device_id": device_id,
        "version": version,
        "timestamp": "",
    }


def discover_samsung_phase_records(
    data_root: Path,
    subjects: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    phases: Optional[List[str]] = None,
    versions: Optional[List[str]] = None,
) -> List[AudioPhaseRecord]:
    subject_filter = {str(item).strip() for item in subjects or [] if str(item).strip()}
    task_filter = {str(item).strip() for item in tasks or [] if str(item).strip()}
    phase_filter = {str(item).strip().lower() for item in phases or [] if str(item).strip()}
    version_filter = {str(item).strip().lower() for item in versions or [] if str(item).strip()}
    records: List[AudioPhaseRecord] = []
    if not data_root.exists():
        return records

    for subject_dir in sorted(data_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_bucket = subject_dir.name
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            meta = parse_samsung_session(session_dir.name, subject_bucket)
            candidates = {subject_bucket, meta["folder_subject"], meta["subject_code"], meta["device_id"]}
            if subject_filter and not candidates.intersection(subject_filter):
                continue
            if version_filter and meta["version"] not in version_filter:
                continue
            for task_dir in sorted(session_dir.iterdir()):
                if not task_dir.is_dir() or not TASK_NAME_RE.match(task_dir.name):
                    continue
                if task_filter and task_dir.name not in task_filter:
                    continue
                for phase_dir in sorted(task_dir.iterdir()):
                    if not phase_dir.is_dir():
                        continue
                    phase_name = phase_dir.name.lower()
                    if phase_name in EXCLUDED_PHASES:
                        continue
                    if phase_filter and phase_name not in phase_filter:
                        continue
                    records.append(
                        AudioPhaseRecord(
                            subject=meta["subject"],
                            folder_subject=meta["folder_subject"],
                            subject_code=meta["subject_code"],
                            device_id=meta["device_id"],
                            version=meta["version"],
                            timestamp=meta["timestamp"],
                            task=task_dir.name,
                            phase=phase_dir.name,
                            phase_dir=phase_dir,
                        )
                    )
    return records


def discover_audio_phase_records(
    data_root: Path,
    subjects: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    phases: Optional[List[str]] = None,
    versions: Optional[List[str]] = None,
    timestamp: Optional[str] = None,
    assessment_phase: Optional[str] = None,
) -> List[Any]:
    records = discover_phase_records(
        data_root=data_root,
        subjects=subjects,
        tasks=tasks,
        phases=phases,
        include_t001=False,
        versions=versions,
        timestamp=timestamp,
        assessment_phase=assessment_phase,
    )
    if records:
        return records
    return discover_samsung_phase_records(
        data_root=data_root,
        subjects=subjects,
        tasks=tasks,
        phases=phases,
        versions=versions,
    )


def leading_trim_seconds(task: Any, phase: Any) -> float:
    key = (str(task or "").strip().upper(), str(phase or "").strip().lower())
    return AUDIO_LEADING_TRIM_SECONDS.get(key, 0.0)


def transcode_audio_to_wav(
    source_path: Path,
    wav_path: Path,
    trim_start_seconds: float = 0.0,
) -> bool:
    """Create the canonical mono 16 kHz WAV after task-specific leading trim."""
    ensure_dir(wav_path.parent)
    cmd = [
        ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
    ]
    if trim_start_seconds > 0:
        cmd.extend(["-ss", f"{trim_start_seconds:g}"])
    cmd.extend([
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        "-y",
        str(wav_path),
    ])
    try:
        subprocess.run(cmd, check=True)
        valid = False
        if wav_path.exists() and wav_path.stat().st_size > 44:
            try:
                with wave.open(str(wav_path), "rb") as wav_file:
                    valid = wav_file.getnframes() > 0
            except (EOFError, wave.Error):
                valid = False
        if not valid:
            wav_path.unlink(missing_ok=True)
        return valid
    except Exception:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def extract_audio_from_video(
    video_path: Path,
    wav_path: Path,
    trim_start_seconds: float = 0.0,
) -> bool:
    return transcode_audio_to_wav(video_path, wav_path, trim_start_seconds)


def extract_audio_wavs(
    data_root: Path,
    wav_root: Path,
    subjects: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    target_tasks: Optional[List[str]] = None,
    phases: Optional[List[str]] = None,
    versions: Optional[List[str]] = None,
    timestamp: Optional[str] = None,
    assessment_phase: Optional[str] = None,
    overwrite: bool = False,
    logger: Optional[StageLogger] = None,
) -> pd.DataFrame:
    ensure_dir(wav_root)
    records = discover_audio_phase_records(
        data_root=data_root,
        subjects=subjects,
        tasks=tasks,
        phases=phases,
        versions=versions,
        timestamp=timestamp,
        assessment_phase=assessment_phase,
    )
    if target_tasks:
        target_task_set = set(target_tasks)
        records = [record for record in records if record.task in target_task_set]
    if logger:
        logger.log(f"[AUDIO WAV] candidate phases={len(records)} target_tasks={target_tasks or 'all'}")
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        source_audio = record.phase_dir / "audio.wav"
        video = record.phase_dir / "video.mp4"
        if not source_audio.exists() and not video.exists():
            continue
        wav_path = wav_root / record.folder_subject / record.task / record.phase / "audio.wav"
        trim_seconds = leading_trim_seconds(record.task, record.phase)
        if PIPELINE_VERBOSE:
            print(
                f"[audio-wav] {idx}/{len(records)} {record.folder_subject}/{record.task}/{record.phase} "
                f"trim_start={trim_seconds:g}s"
            )
        # Always rebuild task-trimmed clips so a legacy untrimmed WAV can
        # never be reused after this preprocessing change.
        if wav_path.exists() and not overwrite and trim_seconds <= 0:
            ok = True
        elif source_audio.exists():
            ok = transcode_audio_to_wav(source_audio, wav_path, trim_seconds)
        else:
            ok = extract_audio_from_video(video, wav_path, trim_seconds)
        rows.append(
            {
                "subject": record.subject,
                "folder_subject": record.folder_subject,
                "subject_code": record.subject_code,
                "device_id": record.device_id,
                "version": record.version,
                "timestamp": record.timestamp,
                "task": record.task,
                "phase": record.phase,
                "video_path": str(video.resolve()) if video.exists() else "",
                "source_audio_path": str(source_audio.resolve()) if source_audio.exists() else "",
                "wav_path": str(wav_path.resolve()),
                "trim_start_seconds": trim_seconds,
                "audio_preprocess_version": AUDIO_PREPROCESS_VERSION,
                "audio_extract_ok": int(ok),
            }
        )
    df = pd.DataFrame(rows)
    manifest_csv = wav_root.parent / "audio_features" / "audio_wav_manifest.csv"
    ensure_dir(manifest_csv.parent)
    df.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    if logger:
        ok_count = int(df.get("audio_extract_ok", pd.Series(dtype=int)).sum()) if not df.empty else 0
        logger.log(f"[AUDIO WAV] saved manifest={manifest_csv} rows={len(df)} ok={ok_count}")
    return df


def extract_audio_features(
    data_root: Path,
    output_dir: Path,
    subjects: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    target_tasks: Optional[List[str]] = None,
    phases: Optional[List[str]] = None,
    versions: Optional[List[str]] = None,
    timestamp: Optional[str] = None,
    assessment_phase: Optional[str] = None,
    overwrite: bool = False,
    retry_failed: bool = False,
    logger: Optional[StageLogger] = None,
) -> pd.DataFrame:
    ensure_dir(output_dir)
    preprocess_meta_path = output_dir / "audio_preprocess.json"
    audio_csv = output_dir / AUDIO_FEATURE_FILENAME
    preprocess_is_current = False
    if preprocess_meta_path.exists():
        try:
            preprocess_meta = json.loads(preprocess_meta_path.read_text(encoding="utf-8-sig"))
            preprocess_is_current = preprocess_meta.get("version") == AUDIO_PREPROCESS_VERSION
        except Exception:
            preprocess_is_current = False
    if audio_csv.exists() and preprocess_is_current and not overwrite and not retry_failed:
        if logger:
            logger.log(f"[AUDIO] reuse existing feature csv: {audio_csv}")
        return pd.read_csv(audio_csv, low_memory=False)

    with optional_stage(logger, "音频依赖检查"):
        required_modules = ["jieba", "textrank4zh", "librosa", "soundfile", "dolphin", "zhconv", "torch", "scipy"]
        missing = [module for module in required_modules if importlib.util.find_spec(module) is None]
        if missing:
            raise ImportError(
                "Full mpcas_audio extraction requires these missing Python packages: "
                + ", ".join(missing)
                + ". Install them in the active environment before running --run-audio."
            )

    wav_root = output_dir.parent / "audio_wav"
    with optional_stage(logger, "音频WAV抽取"):
        manifest = extract_audio_wavs(
            data_root=data_root,
            wav_root=wav_root,
            subjects=subjects,
            tasks=tasks,
            target_tasks=target_tasks,
            phases=phases,
            versions=versions,
            timestamp=timestamp,
            assessment_phase=assessment_phase,
            overwrite=overwrite,
            logger=logger,
        )
    if manifest.empty or int(manifest.get("audio_extract_ok", pd.Series(dtype=int)).sum()) == 0:
        empty = pd.DataFrame(columns=["subject", "task", "phase"])
        empty.to_csv(audio_csv, index=False, encoding="utf-8-sig")
        if logger:
            logger.log(f"[AUDIO] no extracted wav files; wrote empty csv={audio_csv}")
        return empty

    runtime_root = output_dir.parent.resolve()
    wav_root_abs = wav_root.resolve()
    audio_csv_abs = audio_csv.resolve()
    cache_dirs = {
        "dolphin": runtime_root / "_dolphin_cache",
        "modelscope": runtime_root / "_modelscope_cache",
        "numba": runtime_root / "_numba_cache",
        "tmp": runtime_root / "_runtime_tmp",
    }
    for cache_dir in cache_dirs.values():
        ensure_dir(cache_dir)
    ensure_dir(audio_csv_abs.parent)

    mpcas_audio_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "mpcas_audio" / "batch_extract.py"),
        "--dataset-dir",
        str(wav_root_abs),
        "--result-csv",
        str(audio_csv_abs),
    ]
    if subjects:
        mpcas_audio_cmd.extend(["--subjects", *subjects])
    filter_tasks = target_tasks or tasks
    if filter_tasks:
        mpcas_audio_cmd.extend(["--tasks", *filter_tasks])
    if versions:
        mpcas_audio_cmd.extend(["--versions", *versions])
    if phases:
        mpcas_audio_cmd.extend(["--phases", *phases])
    if retry_failed:
        mpcas_audio_cmd.append("--retry-failed")
    env = {
        "MPCAS_AUDIO_DATASET_DIR": str(wav_root_abs),
        "MPCAS_AUDIO_RESULT_CSV": str(audio_csv_abs),
        "PYTHONIOENCODING": "utf-8",
        "MPCAS_AUDIO_DOLPHIN_MODEL_DIR": os.environ.get("MPCAS_AUDIO_DOLPHIN_MODEL_DIR", str(cache_dirs["dolphin"])),
        "MODELSCOPE_CACHE": str(cache_dirs["modelscope"]),
        "NUMBA_CACHE_DIR": str(cache_dirs["numba"]),
        "TMP": str(cache_dirs["tmp"]),
        "TEMP": str(cache_dirs["tmp"]),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        "MPCAS_AUDIO_DOLPHIN_MODEL": os.environ.get("MPCAS_AUDIO_DOLPHIN_MODEL", "small"),
        "MPCAS_AUDIO_DOLPHIN_DEVICE": os.environ.get("MPCAS_AUDIO_DOLPHIN_DEVICE", "auto"),
        "MPCAS_AUDIO_DOLPHIN_DECODING": os.environ.get("MPCAS_AUDIO_DOLPHIN_DECODING", "attention"),
        "MPCAS_AUDIO_DOLPHIN_BEAM_SIZE": os.environ.get("MPCAS_AUDIO_DOLPHIN_BEAM_SIZE", "1"),
    }
    if logger:
        env["PIPELINE_STAGE_LOG"] = str(logger.log_path)
    with optional_stage(logger, "音频ASR与完整特征提取"):
        run_command(
            mpcas_audio_cmd,
            cwd=PROJECT_ROOT,
            skip_existing=None,
            env=env,
            logger=logger,
        )
    with optional_stage(logger, "音频特征表加载"):
        result = pd.read_csv(audio_csv, low_memory=False)
        preprocess_meta_path.write_text(
            json.dumps(
                {
                    "version": AUDIO_PREPROCESS_VERSION,
                    "leading_trim_seconds": {
                        f"{task}/{phase}": seconds
                        for (task, phase), seconds in AUDIO_LEADING_TRIM_SECONDS.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return result
