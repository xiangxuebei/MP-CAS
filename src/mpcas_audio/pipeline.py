"""One-record speech feature extraction and result serialization."""
import csv
import io
import os
import numpy as np
import librosa
import torch
from .context import infer_task_context, extract_path_fields
from .asr import transcribe_with_dolphin
from .acoustic import *
from .language import *
RESULT_DIR = os.environ.get("MPCAS_AUDIO_RESULT_DIR", "outputs/audio_language")
RESULT_CSV_PATH = os.environ.get("MPCAS_AUDIO_RESULT_CSV", os.path.join(RESULT_DIR, "audio_features.csv"))


def extract_audio_features(audio_path):
    """提取语音及语言特征。"""
    try:
        is_m4a = audio_path.lower().endswith(".m4a")
        task_context = infer_task_context(audio_path)
        expected_reference = "，".join(task_context.get("target_words", [])) if task_context.get("target_words") else ("图片讲述（无固定参考文本）" if task_context.get("task_type") == "story" else "")

        # 音频加载与预处理
        if is_m4a:
            wav_data = m4a_to_wav(audio_path)
            if not wav_data:
                return get_error_result(audio_path, "m4a转换失败", expected_reference)
            y, sr = librosa.load(io.BytesIO(wav_data), sr=22050, mono=True)
        else:
            y, sr = librosa.load(audio_path, sr=22050, mono=True)

        # 前置低通滤波，清理高频噪声
        y = apply_lowpass_filter(y, sr)
        
        # 清理音频数据中的异常值
        y = clean_array(y, 0.0, -1.0, 1.0)
        
        # 音频预处理：降噪+归一化（使用兼容函数）
        y = librosa.effects.trim(y, top_db=20)[0]  # 去除首尾静音
        y = audio_normalize(y)  # 兼容新旧版本librosa

        # 语音识别
        audio_text = transcribe_with_dolphin(audio_path, prompt_text=task_context.get("prompt_text", ""))

        # 文本分析
        task_text_analysis = analyze_task_text(audio_text, task_context)

        # 基础时长信息
        duration = librosa.get_duration(y=y, sr=sr)
        duration = clean_numeric_value(duration, 5.0, 0.5, None)
        hop_length = 512

        # 停顿相关特征（统一检测逻辑）
        pause_ratio, pause_total_duration, pause_freq_per_min, avg_pause_dur = calculate_pause_metrics(y, sr, hop_length)
        effective_duration = duration - pause_total_duration
        effective_duration = clean_numeric_value(effective_duration, 0.5, 0.0, duration)

        # 音调特征
        avg_f0, f0_std, f0_range, pitch_jitter = calculate_f0_features(y, sr)
        average_pitch = avg_f0  # 平均音高=平均基频

        # 谐噪比
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        y_trimmed = clean_array(y_trimmed, 0.0, -1.0, 1.0)
        
        if len(y_trimmed) > 0:
            harmonic, percussive = librosa.effects.hpss(y_trimmed)
            harmonic = clean_array(harmonic, 0.0, -1.0, 1.0)
            percussive = clean_array(percussive, 0.0, -1.0, 1.0)
            
            harmonic_energy = np.sum(harmonic**2)
            harmonic_energy = clean_numeric_value(harmonic_energy, 0.1, 0.001, None)
            
            noise_energy = np.sum((y_trimmed - harmonic)**2)
            noise_energy = clean_numeric_value(noise_energy, 0.01, 0.0, None)
            
            hnr = 10 * np.log10(harmonic_energy / (noise_energy + 1e-10)) if noise_energy > 0 else 0
            hnr = clean_numeric_value(hnr, 20.0, -10.0, 50.0)
        else:
            hnr = 0.0

        # 其他特征
        vibrato_freq, vibrato_amplitude = extract_vibrato_features(y, sr, hop_length)
        info_recall, info_precision, info_f1 = calculate_info_metrics(
            audio_text, "", task_context.get("target_words")
        )
        semantic_sim = task_text_analysis.get("semantic_similarity", 0) if task_text_analysis else 0
        semantic_sim = clean_numeric_value(semantic_sim, 50.0, 0.0, 100.0)
        
        speech_rate, speaking_rate = calculate_speech_metrics(audio_text, duration, effective_duration)
        avg_sentence_len = calculate_avg_sentence_length(audio_text)
        avg_volume, volume_std = calculate_volume_features(y, sr)

        # 共振峰特征
        f1, f2, f3, f1_bw, f2_bw, f3_bw = calculate_formant_features(y, sr)
        phone_duration_std, phone_duration_cv = calculate_phone_duration_variation(y, sr, audio_text)
        rhythm_strength, intonation_slope, accent_freq = calculate_prosodic_features(y, sr)
        _, avg_word_length, syntactic_complexity = calculate_language_complexity(audio_text)
        spectral_entropy, energy_entropy = calculate_sound_entropy(y, sr)
        spec_flatness, timbre_change, freq_coverage = calculate_sound_diversity(y, sr)

        # 新增文本特征
        ttr, mtld_norm, entropy_norm, rare_word_ratio = calculate_lexical_diversity(audio_text)
        filler_freq_total, filler_freq_by_type, repeat_freq, repeat_ratio = calculate_filler_repeat_freq(
            text=audio_text, audio_duration=duration
        )
        semantic_pause_rate, grammar_pause_ratio, avg_semantic_pause, position_deviation = calculate_semantic_pause_metrics(
            y, sr, audio_text
        )
        subject, task, phase = extract_path_fields(audio_path)

        # 清理GPU内存
        torch.cuda.empty_cache()

        # 构建结果
        return {
            "subject": subject,
            "task": task,
            "phase": phase,
            "文件名": os.path.basename(audio_path),
            "时长(秒)": round(duration, 2),
            "有效语音时长(秒)": effective_duration,
            "停顿总时长(秒)": pause_total_duration,
            "平均停顿时长(秒)": avg_pause_dur,
            "停顿频率(次/分钟)": pause_freq_per_min,
            "停顿比例(%)": pause_ratio,
            "平均音高(Hz)": round(average_pitch, 2),
            "平均基频(Hz)": avg_f0,
            "基频标准差(Hz)": f0_std,
            "基频范围(Hz)": f0_range,
            "抖动(Hz)": round(pitch_jitter, 2),
            "谐噪比(dB)": round(hnr, 2),
            "平均音量(振幅)": avg_volume,
            "音量标准差": volume_std,
            "颤动频率(Hz)": vibrato_freq,
            "颤动幅度(Hz)": vibrato_amplitude,
            "语速(字/分钟)": speech_rate,
            "说话率(字/分钟)": speaking_rate,
            "平均句子长度(字)": avg_sentence_len,
            
            # 共振峰特征
            "共振峰F1(Hz)": f1,
            "共振峰F2(Hz)": f2,
            "共振峰F3(Hz)": f3,
            "F1带宽(Hz)": f1_bw,
            "F2带宽(Hz)": f2_bw,
            "F3带宽(Hz)": f3_bw,
            
            # “音素”（语音段）时长特征
            "音素时长标准差(秒)": phone_duration_std,
            "音素时长变异系数": phone_duration_cv,
            
            # 韵律特征
            "节奏强度": rhythm_strength,
            "语调斜率(Hz/秒)": intonation_slope,
            "重音频率(个/秒)": accent_freq,
            
            # 语言复杂度
            "词汇多样性-TTR": ttr,
            "词汇多样性-MTLD": mtld_norm,
            "词汇多样性-熵值": entropy_norm,
            "稀有词占比": rare_word_ratio,
            "平均词长(字符)": avg_word_length,
            "句法复杂度(虚词占比)": syntactic_complexity,
            
            # 填充词与重复词特征
            "总填充词频率(个/分钟)": filler_freq_total,
            "语气填充词频率(个/分钟)": filler_freq_by_type["语气填充词"],
            "逻辑填充词频率(个/分钟)": filler_freq_by_type["逻辑填充词"],
            "犹豫填充词频率(个/分钟)": filler_freq_by_type["犹豫填充词"],
            "重复词频率(个/分钟)": repeat_freq,
            "重复词占比": repeat_ratio,
            
            # 语义停顿特征
            "语义停顿率(%)": semantic_pause_rate,
            "语法停顿占比(%)": grammar_pause_ratio,
            "平均语义停顿时长(秒)": avg_semantic_pause,
            "停顿位置偏差度(秒)": position_deviation,
            
            # 声音熵与多样性
            "频谱熵(归一化)": spectral_entropy,
            "能量熵(归一化)": energy_entropy,
            "频谱平坦度": spec_flatness,
            "音色变化率": timbre_change,
            "频率覆盖范围(归一化)": freq_coverage,

            "识别文字": audio_text,
            "原文本": expected_reference,
            "语义相似度(%)": task_text_analysis.get("semantic_similarity", semantic_sim) if task_text_analysis else semantic_sim,
            "事实一致性(%)": task_text_analysis.get("factual_consistency", 0) if task_text_analysis else 0,
            "叙事连贯分数(%)": task_text_analysis.get("narrative_coherence", 0) if task_text_analysis else 0,
            "信息完整性(%)": task_text_analysis.get("information_completeness", 0) if task_text_analysis else 0,
            "时间顺序一致性(%)": task_text_analysis.get("temporal_consistency", 0) if task_text_analysis else 0,
            "信息单元召回率(%)": task_text_analysis.get("info_recall_rate", info_recall) if task_text_analysis else info_recall,
            "信息单元精确率(%)": task_text_analysis.get("info_precision_rate", info_precision) if task_text_analysis else info_precision,
            "信息单元F1(%)": task_text_analysis.get("overall_quality", info_f1) if task_text_analysis else info_f1,
            "整体质量评分(%)": task_text_analysis.get("overall_quality", 0) if task_text_analysis else 0,
            "错误分析": task_text_analysis.get("error_analysis", "") if task_text_analysis else "",
            "纠错建议": task_text_analysis.get("correction_suggestions", "") if task_text_analysis else "",
            "错误信息": ""
        }
    except Exception as e:
        torch.cuda.empty_cache()
        return get_error_result(audio_path, f"处理失败: {str(e)}", expected_reference)


def get_error_result(filename, error_msg, reference_text=""):
    """错误结果处理"""
    subject, task, phase = extract_path_fields(filename)
    error_dict = {
        "subject": subject,
        "task": task,
        "phase": phase,
        "文件名": filename,
        "错误信息": error_msg,
        "时长(秒)": 0,
        "有效语音时长(秒)": 0,
        "停顿总时长(秒)": 0,
        "平均停顿时长(秒)": 0,
        "停顿频率(次/分钟)": 0,
        "停顿比例(%)": 0,
        "平均音高(Hz)": 0,
        "平均基频(Hz)": 0,
        "基频标准差(Hz)": 0,
        "基频范围(Hz)": 0,
        "抖动(Hz)": 0,
        "谐噪比(dB)": 0,
        "平均音量(振幅)": 0,
        "音量标准差": 0,
        "颤动频率(Hz)": 0,
        "颤动幅度(Hz)": 0,
        "语速(字/分钟)": 0,
        "说话率(字/分钟)": 0,
        "平均句子长度(字)": 0,
        
        # 共振峰特征
        "共振峰F1(Hz)": 0, "共振峰F2(Hz)": 0, "共振峰F3(Hz)": 0,
        "F1带宽(Hz)": 0, "F2带宽(Hz)": 0, "F3带宽(Hz)": 0,
        
        # 音素时长特征
        "音素时长标准差(秒)": 0, "音素时长变异系数": 0,
        
        # 韵律特征
        "节奏强度": 0, "语调斜率(Hz/秒)": 0, "重音频率(个/秒)": 0,
        
        # 语言复杂度
        "词汇多样性-TTR": 0, "词汇多样性-MTLD": 0, "词汇多样性-熵值": 0,
        "稀有词占比": 0, "平均词长(字符)": 0, "句法复杂度(虚词占比)": 0,
        
        # 填充词与重复词特征
        "总填充词频率(个/分钟)": 0, "语气填充词频率(个/分钟)": 0,
        "逻辑填充词频率(个/分钟)": 0, "犹豫填充词频率(个/分钟)": 0,
        "重复词频率(个/分钟)": 0, "重复词占比": 0,
        
        # 语义停顿特征
        "语义停顿率(%)": 0, "语法停顿占比(%)": 0,
        "平均语义停顿时长(秒)": 0, "停顿位置偏差度(秒)": 0,
        
        # 声音熵与多样性
        "频谱熵(归一化)": 0, "能量熵(归一化)": 0,
        "频谱平坦度": 0, "音色变化率": 0, "频率覆盖范围(归一化)": 0,

        "识别文字": "", "原文本": reference_text,
        "语义相似度(%)": 0, "事实一致性(%)": 0, "叙事连贯分数(%)": 0,
        "信息完整性(%)": 0, "时间顺序一致性(%)": 0, "信息单元召回率(%)": 0,
        "信息单元精确率(%)": 0, "信息单元F1(%)": 0, "整体质量评分(%)": 0,
        "错误分析": "", "纠错建议": ""
    }
    return error_dict


def write_results_to_csv(results_list):
    """写入CSV结果"""
    fieldnames = [
        "subject",
        "task",
        "phase",
        "文件名", "时长(秒)", "有效语音时长(秒)", "停顿总时长(秒)", 
        "平均停顿时长(秒)", "停顿频率(次/分钟)", "停顿比例(%)",
        "平均音高(Hz)", "平均基频(Hz)",
        "基频标准差(Hz)", "基频范围(Hz)", "抖动(Hz)", "谐噪比(dB)",
        "平均音量(振幅)", "音量标准差", "颤动频率(Hz)", "颤动幅度(Hz)",
        "语速(字/分钟)", "说话率(字/分钟)", "平均句子长度(字)",
        
        # 共振峰特征
        "共振峰F1(Hz)", "共振峰F2(Hz)", "共振峰F3(Hz)",
        "F1带宽(Hz)", "F2带宽(Hz)", "F3带宽(Hz)",
        
        # 音素时长特征
        "音素时长标准差(秒)", "音素时长变异系数",
        
        # 韵律特征
        "节奏强度", "语调斜率(Hz/秒)", "重音频率(个/秒)",
        
        # 语言复杂度
        "词汇多样性-TTR", "词汇多样性-MTLD", "词汇多样性-熵值",
        "稀有词占比", "平均词长(字符)", "句法复杂度(虚词占比)",
        
        # 填充词与重复词特征
        "总填充词频率(个/分钟)", "语气填充词频率(个/分钟)",
        "逻辑填充词频率(个/分钟)", "犹豫填充词频率(个/分钟)",
        "重复词频率(个/分钟)", "重复词占比",
        
        # 语义停顿特征
        "语义停顿率(%)", "语法停顿占比(%)",
        "平均语义停顿时长(秒)", "停顿位置偏差度(秒)",
        
        # 声音熵与多样性
        "频谱熵(归一化)", "能量熵(归一化)",
        "频谱平坦度", "音色变化率", "频率覆盖范围(归一化)",
        
        "识别文字", "原文本", "语义相似度(%)", "事实一致性(%)",
        "叙事连贯分数(%)", "信息完整性(%)", "时间顺序一致性(%)",
        "信息单元召回率(%)", "信息单元精确率(%)", "信息单元F1(%)",
        "整体质量评分(%)", "错误分析", "纠错建议", "错误信息"
    ]
    
    with open(RESULT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results_list:
            # 补充缺失的字段，避免KeyError
            for field in fieldnames:
                if field not in result:
                    result[field] = 0 if field.endswith((')', '秒', '字', 'Hz', 'dB', '%')) else ""
            writer.writerow(result)
