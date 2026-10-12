"""Speech timing, pauses, sentence rate and phone-duration features."""
import re
import numpy as np
import librosa
from .signal import clean_array, clean_numeric_value


def detect_pause_segments(y, sr, hop_length=512, db_threshold=-40.0, min_pause_duration=0.25):
    """
    使用统一的能量阈值检测停顿段：
    - 先计算 RMS -> dB，相对最大值（最大帧为 0 dB）
    - 低于 db_threshold 且持续时间 >= min_pause_duration 的片段视为停顿
    返回: (pause_segments, total_duration)
        pause_segments: [(start_time, duration), ...]
    """
    total_duration = librosa.get_duration(y=y, sr=sr)
    total_duration = clean_numeric_value(total_duration, 1.0, 0.1, None)

    if total_duration == 0:
        return [], 0.0

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms = clean_array(rms, 0.001, 0.0, None)

    if len(rms) == 0:
        return [], total_duration

    # 相对 dB：最大能量帧 = 0 dB
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    pause_frames = np.where(rms_db < db_threshold)[0]

    pause_segments = []
    if len(pause_frames) > 0:
        start_f = pause_frames[0]
        prev_f = pause_frames[0]
        for f in pause_frames[1:]:
            if f == prev_f + 1:
                prev_f = f
            else:
                start_time = start_f * hop_length / sr
                end_time = (prev_f + 1) * hop_length / sr
                dur = end_time - start_time
                if dur >= min_pause_duration:
                    pause_segments.append((start_time, dur))
                start_f = prev_f = f
        # 最后一个片段
        start_time = start_f * hop_length / sr
        end_time = (prev_f + 1) * hop_length / sr
        dur = end_time - start_time
        if dur >= min_pause_duration:
            pause_segments.append((start_time, dur))

    return pause_segments, total_duration


def calculate_semantic_pause_metrics(y, sr, text, hop_length=512):
    """
    语义停顿指标：区分"语法停顿"与"非语法停顿"
    返回：
      语义停顿率(%)              -> 非语法停顿时长 / 总时长
      语法停顿占比(%)            -> 语法停顿次数 / 全部停顿次数
      平均语义停顿时长(秒)       -> 非语法停顿平均时长
      停顿位置偏差度(秒)         -> 语法停顿中心与标点位置的平均时间差
    """
    if not text.strip():
        return 0.0, 0.0, 0.0, 0.0
    
    # 1. 基于统一逻辑检测所有停顿
    pause_segments, total_duration = detect_pause_segments(y, sr, hop_length=hop_length)
    total_duration = clean_numeric_value(total_duration, 1.0, 0.1, None)

    if total_duration <= 0 or len(pause_segments) == 0:
        return 0.0, 0.0, 0.0, 0.0

    total_pause_duration = sum(d for _, d in pause_segments)

    # 2. 文本层面：确定语法停顿位置（标点处）
    punctuation_positions = []
    punc_pattern = re.compile(r"[。，！？；,.!?;]")
    for match in punc_pattern.finditer(text):
        char_pos = match.start() / len(text) if len(text) > 0 else 0
        estimated_time = char_pos * total_duration
        punctuation_positions.append(estimated_time)
    
    # 3. 匹配语法停顿（停顿中心在标点±500ms范围内）
    grammar_pause_count = 0
    grammar_pause_duration = 0.0
    for start_time, duration in pause_segments:
        pause_center = start_time + duration / 2
        for punc_time in punctuation_positions:
            if abs(pause_center - punc_time) < 0.5:  # 500ms阈值
                grammar_pause_count += 1
                grammar_pause_duration += duration
                break
    
    # 语法/语义停顿拆分
    grammar_pause_ratio = (grammar_pause_count / len(pause_segments)) * 100 if len(pause_segments) > 0 else 0.0
    grammar_pause_ratio = clean_numeric_value(grammar_pause_ratio, 30.0, 0.0, 100.0)

    semantic_pause_duration = max(total_pause_duration - grammar_pause_duration, 0.0)
    semantic_pause_count = max(len(pause_segments) - grammar_pause_count, 0)

    semantic_pause_rate = (semantic_pause_duration / total_duration) * 100 if total_duration > 0 else 0.0
    semantic_pause_rate = clean_numeric_value(semantic_pause_rate, 5.0, 0.0, 100.0)
    
    avg_semantic_pause = semantic_pause_duration / semantic_pause_count if semantic_pause_count > 0 else 0.0
    avg_semantic_pause = clean_numeric_value(avg_semantic_pause, 0.5, 0.0, 5.0)
    
    # 停顿位置偏差度（语法停顿与标点的平均时间差）
    position_deviation = 0.0
    matched_count = 0
    for start_time, duration in pause_segments:
        pause_center = start_time + duration / 2
        for punc_time in punctuation_positions:
            if abs(pause_center - punc_time) < 0.5:
                position_deviation += abs(pause_center - punc_time)
                matched_count += 1
                break
    position_deviation = position_deviation / matched_count if matched_count > 0 else 0.0
    position_deviation = clean_numeric_value(position_deviation, 0.2, 0.0, 1.0)
    
    return (round(semantic_pause_rate, 2),
            round(grammar_pause_ratio, 2),
            round(avg_semantic_pause, 4),
            round(position_deviation, 4))


def calculate_speech_metrics(text, total_duration, effective_duration):
    valid_chars = re.findall(r"[\u4e00-\u9fa5]", text)
    valid_char_count = len(valid_chars)
    
    speech_rate = (valid_char_count / effective_duration) * 60 if effective_duration > 0 else 0.0
    speech_rate = clean_numeric_value(speech_rate, 150.0, 30.0, 300.0)
    
    speaking_rate = (valid_char_count / total_duration) * 60 if total_duration > 0 else 0.0
    speaking_rate = clean_numeric_value(speaking_rate, 120.0, 30.0, 300.0)
    
    return round(speech_rate, 2), round(speaking_rate, 2)


def calculate_pause_metrics(y, sr, hop_length=512):
    """
    基本停顿特征（修正版）：
    - 使用 detect_pause_segments 统一检测停顿
    - 去掉录音最开始/最后 300ms 内的停顿，避免启动/结束静音干扰
    - 返回：
        停顿比例(%), 停顿总时长(秒), 停顿频率(次/分钟), 平均停顿时长(秒)
    """
    pause_segments, total_duration = detect_pause_segments(y, sr, hop_length=hop_length)
    total_duration = clean_numeric_value(total_duration, 1.0, 0.1, None)

    if total_duration == 0:
        return 0.0, 0.0, 0.0, 0.0

    # 去掉录音首尾的停顿（认为不属于“说话中的犹豫”）
    INTERNAL_MARGIN = 0.3  # 300ms
    internal_segments = []
    for start, dur in pause_segments:
        end = start + dur
        if start < INTERNAL_MARGIN:
            continue
        if end > total_duration - INTERNAL_MARGIN:
            continue
        internal_segments.append((start, dur))

    if internal_segments:
        pause_segments = internal_segments

    if not pause_segments:
        return 0.0, 0.0, 0.0, 0.0

    pause_total_duration = sum(d for _, d in pause_segments)
    pause_total_duration = clean_numeric_value(pause_total_duration, 0.0, 0.0, total_duration)

    pause_ratio = (pause_total_duration / total_duration) * 100 if total_duration > 0 else 0.0
    pause_ratio = clean_numeric_value(pause_ratio, 5.0, 0.0, 100.0)

    pause_count = len(pause_segments)
    pause_freq_per_min = (pause_count / total_duration) * 60 if total_duration > 0 else 0.0
    pause_freq_per_min = clean_numeric_value(pause_freq_per_min, 0.0, 0.0, 200.0)

    avg_pause_duration = pause_total_duration / pause_count if pause_count > 0 else 0.0
    # 平均停顿时长更关注“思考停顿”，这里允许上限大一些
    avg_pause_duration = clean_numeric_value(avg_pause_duration, 0.6, 0.0, 6.0)

    return (round(pause_ratio, 2),
            round(pause_total_duration, 2),
            round(pause_freq_per_min, 2),
            round(avg_pause_duration, 2))


def calculate_avg_sentence_length(text):
    valid_chars = re.findall(r"[\u4e00-\u9fa5]", text)
    valid_char_count = len(valid_chars)
    if valid_char_count == 0:
        return 5.0  # 兜底默认值
    
    # 尝试用标点分割
    sentence_separators = r"[。！？；，\n、,.!?;]"
    sentences = re.split(sentence_separators, text)
    valid_sentences = [s for s in sentences if re.search(r"[\u4e00-\u9fa5]", s.strip())]
    
    # 若无有效句子（无标点），按10字分割
    if len(valid_sentences) < 1:
        sentence_count = max(1, len(text) // 10)
    else:
        sentence_count = len(valid_sentences)
    
    avg_sentence_len = valid_char_count / sentence_count
    avg_sentence_len = clean_numeric_value(avg_sentence_len, 10.0, 1.0, 30.0)
    
    return round(avg_sentence_len, 2)


def calculate_effective_speech_duration(y, sr, hop_length=512):
    """
    备用函数：基于统一停顿检测计算有效语音时长，可在外部单独调用。
    在 extract_audio_features 中我们直接用 (总时长 - 停顿总时长)。
    """
    pause_segments, total_duration = detect_pause_segments(y, sr, hop_length=hop_length)
    total_duration = clean_numeric_value(total_duration, 1.0, 0.1, None)
    if total_duration == 0:
        return 0.0
    pause_total_duration = sum(d for _, d in pause_segments)
    pause_total_duration = clean_numeric_value(pause_total_duration, 0.0, 0.0, total_duration)
    effective_duration = total_duration - pause_total_duration
    effective_duration = clean_numeric_value(effective_duration, 0.5, 0.0, total_duration)
    return round(effective_duration, 2)


def calculate_phone_duration_variation(y, sr, text, hop_length=512):
    """
    语音段时长变异（更贴近节奏）：
    - 基于停顿反推语音段
    - 统计语音段时长的标准差和变异系数
    """
    pause_segments, total_duration = detect_pause_segments(y, sr, hop_length=hop_length)
    total_duration = clean_numeric_value(total_duration, 1.0, 0.1, None)

    if total_duration == 0:
        return 0.0, 0.0

    # 语音段（非停顿部分）
    speech_segments = []
    if not pause_segments:
        speech_segments = [(0.0, total_duration)]
    else:
        current = 0.0
        for start, dur in pause_segments:
            if start > current:
                speech_segments.append((current, start - current))
            current = start + dur
        if current < total_duration:
            speech_segments.append((current, total_duration - current))

    # 过滤特别短的语音段（<200ms）
    speech_durations = [dur for _, dur in speech_segments if dur >= 0.2]

    if len(speech_durations) < 2:
        # 只有一段连续语音：节奏非常均匀，给一个很低的变异度而不是全 0
        return 0.05, 0.1

    speech_durations = np.array(speech_durations)
    speech_durations = clean_array(speech_durations, 0.2, 0.1, 8.0)

    duration_std = np.std(speech_durations)
    duration_std = clean_numeric_value(duration_std, 0.2, 0.0, 4.0)
    duration_mean = np.mean(speech_durations)
    duration_mean = clean_numeric_value(duration_mean, 0.8, 0.1, 8.0)

    duration_cv = duration_std / duration_mean if duration_mean > 0 else 0.0
    duration_cv = clean_numeric_value(duration_cv, 0.25, 0.0, 2.0)

    return round(duration_std, 4), round(duration_cv, 4)
