"""Join OpenFace AU intensities with Py-Feat ResMaskNet emotion probabilities."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


EMOTION_NAMES = ("Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger")


def read_frame_csv(path: Path) -> pd.DataFrame:
    result = pd.read_csv(path, low_memory=False)
    if result.empty:
        raise ValueError(f"No frame rows in {path}")
    return result


def _openface_executable(path: Path | None) -> Path:
    candidate = str(path) if path else shutil.which("FeatureExtraction") or shutil.which("FeatureExtraction.exe")
    if not candidate:
        raise FileNotFoundError("Pass --openface-exe or add FeatureExtraction to PATH")
    resolved = Path(candidate).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def openface_video(video: Path, executable: Path | None = None) -> pd.DataFrame:
    binary = _openface_executable(executable)
    with TemporaryDirectory(prefix="mpcas_openface_") as work:
        command = [str(binary), "-f", str(video.resolve()), "-out_dir", work, "-of", "features", "-pose", "-aus"]
        result = subprocess.run(command, cwd=binary.parent, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(f"OpenFace failed for {video}: {result.stderr[-1200:]}")
        outputs = list(Path(work).glob("*.csv"))
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one OpenFace CSV, found {len(outputs)} for {video}")
        frame = read_frame_csv(outputs[0])
    frame.columns = frame.columns.astype(str).str.strip()
    if "frame" not in frame.columns:
        raise ValueError("OpenFace output lacks frame column")
    if "success" in frame.columns:
        frame = frame[pd.to_numeric(frame["success"], errors="coerce").eq(1)].copy()
    frame["frame_idx"] = pd.to_numeric(frame["frame"], errors="raise").astype(int) - 1
    if frame.frame_idx.lt(0).any() or frame.frame_idx.duplicated().any():
        raise ValueError("Invalid OpenFace frame indices")
    for axis, source in (("pitch", "pose_Rx"), ("yaw", "pose_Ry"), ("roll", "pose_Rz")):
        if source in frame:
            frame[f"head_{axis}"] = np.rad2deg(pd.to_numeric(frame[source], errors="coerce"))
    keep = ["frame_idx", *[name for name in frame if name.startswith("AU") and name.endswith("_r")],
            *[name for name in ("head_pitch", "head_yaw", "head_roll") if name in frame]]
    return frame[keep]


def resmasknet_video(video: Path, *, device: str = "cpu", batch_size: int = 8) -> pd.DataFrame:
    try:
        from feat import Detectorv1
    except ImportError as exc:
        raise RuntimeError("Py-Feat Detectorv1 requires Python 3.11+ and the face extra") from exc
    detector = Detectorv1(au_model=None, emotion_model="resmasknet", identity_model=None,
                          gaze_model=None, device=device)
    result = detector.detect(str(video), data_type="video", batch_size=batch_size, progress_bar=False)
    if result is None or len(result) == 0:
        raise ValueError(f"ResMaskNet returned no detected faces for {video}")
    frame = pd.DataFrame(result).copy()
    if "frame" not in frame:
        raise ValueError("Py-Feat output lacks frame column")
    frame["frame_idx"] = pd.to_numeric(frame["frame"], errors="raise").astype(int)
    if frame.frame_idx.duplicated().any():
        raise ValueError("Multiple faces in a frame; provide a single-person video")
    lookup = {str(column).lower(): column for column in frame.columns}
    out = pd.DataFrame({"frame_idx": frame.frame_idx})
    for emotion in EMOTION_NAMES:
        source = lookup.get(emotion.lower())
        if source is None:
            raise ValueError(f"ResMaskNet output lacks {emotion} probability")
        out[f"emotion_{emotion.lower()}"] = pd.to_numeric(frame[source], errors="coerce")
    return out


def infer_video(video: Path, *, openface_exe: Path | None = None, device: str = "cpu",
                batch_size: int = 8) -> pd.DataFrame:
    au_pose = openface_video(video, openface_exe)
    emotion = resmasknet_video(video, device=device, batch_size=batch_size)
    merged = au_pose.merge(emotion, on="frame_idx", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(f"OpenFace and ResMaskNet had no matching valid frames: {video}")
    return merged.sort_values("frame_idx").reset_index(drop=True)
