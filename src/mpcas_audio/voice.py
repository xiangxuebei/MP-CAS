"""Pitch, formant, prosodic and spectral features."""
import numpy as np
import librosa
from scipy.signal import find_peaks
from .signal import clean_array, clean_numeric_value


def extract_vibrato_features(y, sr, hop_length=512):
    """
    颤动频率 + 音高波动幅度（更偏“语调变化”，适合老人语音）
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=300, sr=sr, frame_length=2048, hop_length=hop_length
        )
        if voiced_flag is None or not voiced_flag.any():
            valid_f0 = np.array([120.0, 125.0])
        else:
            valid_f0 = f0[voiced_flag]

        valid_f0 = clean_array(valid_f0, 120.0, 50.0, 400.0)

        if len(valid_f0) < 10:
            amp = np.std(valid_f0)
            amp = clean_numeric_value(amp, 3.0, 0.5, 80.0)
            return 5.0, round(float(amp), 4)

        # 绝对幅度（Hz）
        vibrato_amplitude = np.std(valid_f0)
        vibrato_amplitude = clean_numeric_value(vibrato_amplitude, 5.0, 0.5, 80.0)

        # 用差分 + FFT 估计“快速起伏”的主频率（保留原逻辑）
        f0_diff = np.diff(valid_f0)
        f0_diff = clean_array(f0_diff, 0.0, -80.0, 80.0)

        if len(f0_diff) > 5:
            fft_result = np.fft.fft(f0_diff)
            fft_result = clean_array(fft_result, 0.0)

            freqs = np.fft.fftfreq(len(f0_diff), d=hop_length / sr)
            freqs = clean_array(freqs, 0.0, 0.1, 15.0)

            positive_idx = freqs > 0
            if positive_idx.any():
                max_idx = np.argmax(np.abs(fft_result[positive_idx]))
                vibrato_freq = freqs[positive_idx][max_idx]
            else:
                vibrato_freq = 5.0
        else:
            vibrato_freq = 5.0

        vibrato_freq = clean_numeric_value(vibrato_freq, 5.0, 0.1, 15.0)

        return round(float(vibrato_freq), 4), round(float(vibrato_amplitude), 4)

    except Exception as e:
        print(f"⚠️ 颤动特征计算警告: {str(e)}")
        return 5.0, 5.0


def calculate_volume_features(y, sr):
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    rms = clean_array(rms, 0.001, 0.0, None)
    
    avg_volume = np.mean(rms) if len(rms) > 0 else 0.0
    avg_volume = clean_numeric_value(avg_volume, 0.01, 0.0, 1.0)
    
    volume_std = np.std(rms) if len(rms) > 0 else 0.0
    volume_std = clean_numeric_value(volume_std, 0.005, 0.0, 1.0)
    
    return round(avg_volume, 6), round(volume_std, 6)


def calculate_f0_features(y, sr):
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
        valid_f0 = f0[voiced_flag] if voiced_flag.any() else np.array([100.0, 105.0, 110.0])
        valid_f0 = clean_array(valid_f0, 100.0, 50.0, 300.0)
        
        avg_f0 = np.mean(valid_f0)
        avg_f0 = clean_numeric_value(avg_f0, 150.0, 50.0, 300.0)
        
        f0_std = np.std(valid_f0)  # 基频标准差（整体波动）
        f0_std = clean_numeric_value(f0_std, 20.0, 0.1, 100.0)
        
        # 抖动（Jitter）：相邻基频差异的平均值（短期波动）
        if len(valid_f0) > 1:
            f0_diff = np.abs(np.diff(valid_f0))
            f0_diff = clean_array(f0_diff, 1.0, 0.0, 50.0)
            jitter = np.mean(f0_diff)
        else:
            jitter = 1.0  # 兜底值
        jitter = clean_numeric_value(jitter, 1.0, 0.0, 20.0)
        
        f0_range = np.max(valid_f0) - np.min(valid_f0) if len(valid_f0) > 1 else 0.0
        f0_range = clean_numeric_value(f0_range, 50.0, 0.0, 250.0)
        
        return (round(avg_f0, 2), round(f0_std, 2), 
                round(f0_range, 2), round(jitter, 2))
    except Exception as e:
        print(f"⚠️ 基频特征计算警告: {str(e)}")
        return (150.0, 20.0, 50.0, 1.0)


def calculate_formant_features(y, sr, hop_length=512):
    """
    共振峰特征（放松裁剪版本）：
    - 仍然基于 LPC，但放宽 F1/F2/F3 合理范围，避免大量值被压在边界
    - 更适合作为“粗略声道特征”，而不是精确的元音声学分析
    """
    try:
        # 去掉首尾静音
        y_trim, _ = librosa.effects.trim(y, top_db=20)
        y_trim = clean_array(y_trim, 0.0, -1.0, 1.0)

        if len(y_trim) < sr * 0.1:  # 有效语音不足100ms
            return 500.0, 1500.0, 2500.0, 150.0, 200.0, 250.0

        # 取中间一小段，避免开头结尾噪声
        center = len(y_trim) // 2
        win_len = int(0.04 * sr)  # 40ms
        start = max(0, center - win_len // 2)
        seg = y_trim[start:start + win_len]
        seg = clean_array(seg, 0.0, -1.0, 1.0)

        # LPC 阶数
        lpc_order = int(2 + sr / 1000)
        try:
            lpc_coeffs = librosa.lpc(seg, order=lpc_order)
        except Exception:
            lpc_coeffs = librosa.lpc(seg, order=12)

        roots = np.roots(lpc_coeffs)
        valid_roots = [r for r in roots if np.imag(r) > 0]

        if len(valid_roots) < 3:
            # 根太少，直接给兜底值
            return 500.0, 1500.0, 2500.0, 150.0, 200.0, 250.0

        # 计算频率 & 带宽
        formant_freqs = np.array([np.angle(r) * (sr / (2 * np.pi)) for r in valid_roots])
        formant_bws = np.array([-0.5 * sr / (2 * np.pi) * np.log(np.abs(r) + 1e-10) for r in valid_roots])

        formant_freqs = clean_array(formant_freqs, 1000.0, 150.0, 4500.0)
        formant_bws = clean_array(formant_bws, 150.0, 30.0, 800.0)

        valid_mask = (formant_freqs > 150) & (formant_freqs < 4500) & (formant_bws > 30) & (formant_bws < 800)
        formant_freqs = formant_freqs[valid_mask]
        formant_bws = formant_bws[valid_mask]

        if len(formant_freqs) < 3:
            return 500.0, 1500.0, 2500.0, 150.0, 200.0, 250.0

        # 从小到大取前三个
        sorted_idx = np.argsort(formant_freqs)[:3]
        f1, f2, f3 = formant_freqs[sorted_idx]
        f1_bw, f2_bw, f3_bw = formant_bws[sorted_idx]

        # 只做宽范围兜底
        f1 = clean_numeric_value(f1, 500.0, 200.0, 1200.0)
        f2 = clean_numeric_value(f2, 1500.0, 700.0, 3500.0)
        f3 = clean_numeric_value(f3, 2500.0, 1200.0, 4500.0)

        f1_bw = clean_numeric_value(f1_bw, 150.0, 40.0, 600.0)
        f2_bw = clean_numeric_value(f2_bw, 200.0, 40.0, 700.0)
        f3_bw = clean_numeric_value(f3_bw, 250.0, 40.0, 800.0)

        return (round(float(f1), 2), round(float(f2), 2), round(float(f3), 2),
                round(float(f1_bw), 2), round(float(f2_bw), 2), round(float(f3_bw), 2))

    except Exception as e:
        print(f"⚠️ 共振峰特征计算警告: {str(e)}")
        return (500.0, 1500.0, 2500.0, 150.0, 200.0, 250.0)


def calculate_prosodic_features(y, sr, hop_length=512):
    try:
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms = clean_array(rms, 0.001, 0.0, None)
        
        if len(rms) == 0:
            return 0.5, 0.0, 0.5
        
        peak_indices, _ = find_peaks(rms, height=np.percentile(rms, 70), distance=3)
        
        peak_intervals = np.diff(peak_indices * hop_length / sr) if len(peak_indices) > 1 else np.array([0.5])
        peak_intervals = clean_array(peak_intervals, 0.5, 0.1, 2.0)
        
        rhythm_strength = 0.0
        if len(peak_intervals) > 2:
            interval_std = np.std(peak_intervals)
            interval_std = clean_numeric_value(interval_std, 0.2, 0.01, 1.0)
            rhythm_strength = 1.0 / interval_std if interval_std > 0 else 0.0
            rhythm_strength = min(rhythm_strength, 5.0)
        rhythm_strength = clean_numeric_value(rhythm_strength, 2.0, 0.1, 5.0)
        
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
        valid_f0 = f0[voiced_flag] if voiced_flag.any() else np.array([100.0, 105.0, 110.0])
        valid_f0 = clean_array(valid_f0, 100.0, 50.0, 300.0)
        
        intonation_slope = 0.0
        if len(valid_f0) > 5:
            time_points = np.arange(len(valid_f0)) * (hop_length / sr)
            slope, _ = np.polyfit(time_points, valid_f0, 1)
            intonation_slope = slope
        intonation_slope = clean_numeric_value(intonation_slope, 0.0, -10.0, 10.0)
        
        total_duration = librosa.get_duration(y=y, sr=sr)
        total_duration = clean_numeric_value(total_duration, 1.0, 0.1, None)
        
        accent_peaks = peak_indices[np.where(rms[peak_indices] > np.percentile(rms, 85))[0]] if len(peak_indices) > 0 else []
        accent_freq = len(accent_peaks) / total_duration if total_duration > 0 else 0.0
        accent_freq = clean_numeric_value(accent_freq, 1.0, 0.1, 5.0)
        
        return round(rhythm_strength, 4), round(intonation_slope, 4), round(accent_freq, 4)
    except Exception as e:
        print(f"⚠️ 韵律特征计算警告: {str(e)}")
        return 2.0, 0.0, 1.0


def calculate_sound_entropy(y, sr, n_bins=256):
    try:
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_bins)
        mel_spec = clean_array(mel_spec, 0.001, 0.0, None)
        
        mel_power = np.mean(mel_spec, axis=1)
        mel_power = clean_array(mel_power, 0.001, 0.0, None)
        
        mel_sum = np.sum(mel_power)
        mel_sum = clean_numeric_value(mel_sum, 0.1, 0.001, None)
        
        mel_probs = mel_power / mel_sum
        mel_probs = clean_array(mel_probs, 1.0/n_bins, 0.0, 1.0)
        
        spectral_entropy = -np.sum(p * np.log2(p + 1e-10) for p in mel_probs)
        spectral_entropy = spectral_entropy / np.log2(n_bins) if n_bins > 0 else 0.0
        spectral_entropy = clean_numeric_value(spectral_entropy, 0.5, 0.0, 1.0)
        
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        rms = clean_array(rms, 0.001, 0.0, None)
        
        rms_sum = np.sum(rms)
        rms_sum = clean_numeric_value(rms_sum, 0.1, 0.001, None)
        
        energy_probs = rms / rms_sum if rms_sum > 0 else np.zeros_like(rms)
        energy_probs = clean_array(energy_probs, 1.0/len(rms) if len(rms) > 0 else 0.0, 0.0, 1.0)
        
        energy_entropy = -np.sum(q * np.log2(q + 1e-10) for q in energy_probs)
        energy_entropy = energy_entropy / np.log2(len(rms)) if len(rms) > 0 else 0.0
        energy_entropy = clean_numeric_value(energy_entropy, 0.5, 0.0, 1.0)
        
        return round(spectral_entropy, 4), round(energy_entropy, 4)
    except Exception as e:
        print(f"⚠️ 声音熵计算警告: {str(e)}")
        return 0.5, 0.5


def calculate_sound_diversity(y, sr):
    try:
        spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
        spec_flatness = clean_array(spec_flatness, 0.1, 0.0, 1.0)
        avg_flatness = np.mean(spec_flatness) if len(spec_flatness) > 0 else 0.0
        avg_flatness = clean_numeric_value(avg_flatness, 0.1, 0.0, 1.0)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec = clean_array(mel_spec, 0.001, 0.0, None)
        
        mel_delta = librosa.feature.delta(mel_spec)
        mel_delta = clean_array(mel_delta, 0.001, None, None)
        
        timbre_change = np.mean(np.abs(mel_delta)) if mel_delta.size > 0 else 0.0
        if timbre_change > 0:
            timbre_change = np.log1p(timbre_change)
        timbre_change = clean_numeric_value(timbre_change, 0.5, 0.0, 5.0)
        
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
        valid_f0 = f0[voiced_flag] if voiced_flag.any() else np.array([100.0, 200.0])
        valid_f0 = clean_array(valid_f0, 150.0, 50.0, 300.0)
        
        freq_coverage = 0.0
        if len(valid_f0) > 0:
            f_min, f_max = np.min(valid_f0), np.max(valid_f0)
            freq_coverage = (f_max - f_min) / 300  # 归一化到0-1范围
        freq_coverage = clean_numeric_value(freq_coverage, 0.3, 0.0, 1.0)
        
        return round(avg_flatness, 4), round(timbre_change, 4), round(freq_coverage, 4)
    except Exception as e:
        print(f"⚠️ 声音多样性计算警告: {str(e)}")
        return 0.1, 0.5, 0.3
