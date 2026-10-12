"""Audio loading, cleanup and basic signal preprocessing."""
import io
import os
import re
import subprocess
import tempfile
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from scipy.signal import butter, filtfilt
_original_torchaudio_load = torchaudio.load


def _torchaudio_load_with_soundfile_fallback(uri, *args, **kwargs):
    try:
        return _original_torchaudio_load(uri, *args, **kwargs)
    except Exception as exc:
        if not isinstance(uri, (str, os.PathLike)):
            raise
        try:
            data, sample_rate = sf.read(str(uri), dtype="float32", always_2d=True)
        except Exception:
            raise exc
        frame_offset = int(kwargs.get("frame_offset", 0) or 0)
        num_frames = int(kwargs.get("num_frames", -1) or -1)
        if frame_offset > 0:
            data = data[frame_offset:]
        if num_frames >= 0:
            data = data[:num_frames]
        waveform = torch.from_numpy(np.ascontiguousarray(data.T))
        if kwargs.get("channels_first", True) is False:
            waveform = waveform.transpose(0, 1)
        print(
            f"[AUDIO LOAD] torchaudio decoder unavailable; soundfile fallback "
            f"used for {uri} ({sample_rate} Hz)",
            flush=True,
        )
        return waveform, int(sample_rate)


def clean_numeric_value(value, default=0.0, min_val=None, max_val=None):
    """清理数值，处理NaN和inf，并可选地限制范围"""
    if isinstance(value, (np.ndarray, list)):
        return clean_numeric_value(value[0] if len(value) > 0 else default, default, min_val, max_val)
    
    if not isinstance(value, (int, float, np.number)):
        return default
        
    if np.isnan(value) or np.isinf(value):
        return default
        
    if min_val is not None and value < min_val:
        return min_val
        
    if max_val is not None and value > max_val:
        return max_val
        
    return value


def clean_array(arr, default=0.0, min_val=None, max_val=None):
    """清理数组中的NaN和inf值"""
    if arr is None:
        return np.array([])
        
    cleaned = np.nan_to_num(arr, nan=default, posinf=max_val if max_val is not None else default, neginf=min_val if min_val is not None else default)
    
    if min_val is not None:
        cleaned = np.maximum(cleaned, min_val)
    if max_val is not None:
        cleaned = np.minimum(cleaned, max_val)
        
    return cleaned


def audio_normalize(y):
    """兼容新旧版本librosa的音频归一化函数"""
    try:
        normalized = librosa.effects.normalize(y)
        return clean_array(normalized, min_val=-1.0, max_val=1.0)
    except AttributeError:
        max_val = np.max(np.abs(y)) if len(y) > 0 else 1.0
        if max_val == 0:
            return y
        normalized = y / max_val
        return clean_array(normalized, min_val=-1.0, max_val=1.0)


def apply_lowpass_filter(y, sr, cutoff=8000, order=6):
    """在特征提取前做简单低通滤波，抑制高频噪声。"""
    try:
        if len(y) == 0 or sr <= 0:
            return y
        nyq = 0.5 * sr
        normal_cutoff = min(cutoff, nyq * 0.99) / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        filtered = filtfilt(b, a, y)
        return clean_array(filtered, 0.0, -1.0, 1.0)
    except Exception as e:
        print(f"⚠️ 低通滤波失败，使用原始音频: {str(e)}")
        return y


def m4a_to_wav(m4a_path):
    """M4A转WAV：带ffmpeg检查和备用方案"""
    if subprocess.call(["which", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        print("❌ 未检测到ffmpeg，请安装：sudo apt install ffmpeg")
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
            subprocess.run(
                ["ffmpeg", "-y", "-i", m4a_path, "-ac", "1", "-ar", "16000", temp_wav.name],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
            )
            with open(temp_wav.name, 'rb') as f:
                wav_data = f.read()
        print(f"✅ M4A转WAV成功（大小：{len(wav_data)//1024}KB）")
        return wav_data
    except Exception as e:
        print(f"❌ M4A转WAV失败: {str(e)}")
        # 备用方案：librosa直接读取
        try:
            y, sr = librosa.load(m4a_path, sr=16000, mono=True)
            y = clean_array(y, 0.0, -1.0, 1.0)
            buffer = io.BytesIO()
            sf.write(buffer, y, sr, format='WAV')
            print("✅ 备用方案：librosa读取M4A成功")
            return buffer.getvalue()
        except Exception as e2:
            print(f"❌ 备用方案失败: {str(e2)}")
            return None


torchaudio.load = _torchaudio_load_with_soundfile_fallback
