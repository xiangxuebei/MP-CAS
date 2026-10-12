"""Language, semantic, lexical, and disfluency descriptors."""
import os
import re
from collections import Counter, defaultdict
import jieba
import jieba.posseg as pseg
import numpy as np
from zhconv import convert
from .acoustic import clean_array, clean_numeric_value
jieba_cache_dir = os.path.expanduser("~/.jieba_cache")
os.makedirs(jieba_cache_dir, exist_ok=True)
jieba.dt.cache_file = os.path.join(jieba_cache_dir, "jieba.cache")
FILLER_WORDS = {
    "语气填充词": ["嗯", "呃", "啊", "哦", "呀", "嘛", "呢", "啦", "呗", "啧", "嗨"],
    "逻辑填充词": ["那个", "那个啥", "然后", "就是", "其实", "主要是", "说白了", "换句话说", "总的来说"],
    "犹豫填充词": ["等一下", "让我想想", "不对", "不是", "哦不对", "应该是", "可能是"]
}
ALL_FILLERS = {word for words in FILLER_WORDS.values() for word in words}


def analyze_task_text(asr_text, context):
    """基于任务上下文的轻量文本分析：回忆任务只看目标词命中，讲故事任务不做事实对齐"""
    context = context or {}
    target_words = context.get("target_words", [])
    task_type = context.get("task_type", "unknown")

    raw_text = convert(asr_text.strip(), 'zh-cn') if isinstance(asr_text, str) and asr_text.strip() else ""
    preview = f"{raw_text[:80]}..." if len(raw_text) > 80 else raw_text
    print(f"文本预览: {preview}")

    try:
        if target_words:
            hits = [w for w in target_words if w in raw_text]
            recall = len(hits) / len(target_words) * 100 if target_words else 0.0
            recognized_vocab = set(w for w in jieba.lcut(raw_text) if len(w) >= 2)
            precision = len(hits) / len(recognized_vocab) * 100 if recognized_vocab else recall
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else recall
            semantic_score = recall  # 对于回忆任务，召回率更能反映认知水平
            error_analysis = f"命中词：{hits}" if hits else "未命中目标词"
        elif task_type == "story":
            # 图片讲述任务无参考文本，保持语义指标中性，只输出文本长度作为完整性参考
            recall = precision = f1 = 0.0
            semantic_score = min(len(raw_text) * 2, 100) if raw_text else 0.0
            error_analysis = "自由叙述任务，无需对齐参考文本"
        else:
            recall = precision = f1 = 0.0
            semantic_score = 0.0
            error_analysis = "未识别到任务类型"

        return {
            "semantic_similarity": round(semantic_score, 2),
            "factual_consistency": round(semantic_score, 2),
            "narrative_coherence": round(semantic_score * 0.9, 2),
            "information_completeness": round(recall, 2),
            "temporal_consistency": round(semantic_score * 0.8, 2),
            "info_recall_rate": round(recall, 2),
            "info_precision_rate": round(precision, 2),
            "overall_quality": round(f1, 2),
            "error_analysis": error_analysis,
            "correction_suggestions": ""
        }
    except Exception as e:
        error_msg = f"文本分析失败: {str(e)}"
        print(error_msg)
        return {
            "semantic_similarity": 0,
            "factual_consistency": 0,
            "narrative_coherence": 0,
            "information_completeness": 0,
            "temporal_consistency": 0,
            "info_recall_rate": 0,
            "info_precision_rate": 0,
            "overall_quality": 0,
            "error_analysis": error_msg,
            "correction_suggestions": ""
        }


def calculate_semantic_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    try:
        words1 = set([w for w in jieba.lcut(text1) if len(w) > 1])
        words2 = set([w for w in jieba.lcut(text2) if len(w) > 1])
        if not words1 or not words2:
            return 0.0
        return round(len(words1 & words2) / len(words1 | words2), 4)
    except Exception as e:
        print(f"⚠️ 语义相似度计算警告: {str(e)}")
        return 0.3


def extract_info_units(text):
    if not text:
        return set()
    try:
        info_units = set()
        words = [w for w in jieba.lcut(text) if len(w) > 1]
        info_units.update(words)
        seg_result = pseg.lcut(text)
        entities = [word for word, flag in seg_result if flag in ['nr', 'ns', 'nt']]
        info_units.update(entities)
        numbers = re.findall(r"\d+[元个天日月岁块]", text)
        info_units.update(numbers)
        core_words = re.findall(r"[\u4e00-\u9fa5]{2,}", text)
        info_units.update(core_words)
        return info_units
    except Exception as e:
        print(f"⚠️ 信息单元提取警告: {str(e)}")
        return set(jieba.lcut(text))


def calculate_info_metrics(asr_text, ref_text, target_words=None):
    try:
        if target_words:
            hits = [w for w in target_words if w in asr_text]
            recall = len(hits) / len(target_words) if target_words else 0.0
            recognized_vocab = set(w for w in jieba.lcut(asr_text) if len(w) >= 2)
            precision = len(hits) / len(recognized_vocab) if recognized_vocab else recall
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else recall
            return round(recall * 100, 2), round(precision * 100, 2), round(f1 * 100, 2)

        asr_info = extract_info_units(asr_text)
        ref_info = extract_info_units(ref_text)
        if not ref_info:
            return 0.0, 0.0, 0.0
        recall = len(asr_info & ref_info) / len(ref_info) if asr_info else 0.0
        precision = len(asr_info & ref_info) / len(asr_info) if asr_info else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return round(recall * 100, 2), round(precision * 100, 2), round(f1 * 100, 2)
    except Exception as e:
        print(f"⚠️ 信息指标计算警告: {str(e)}")
        return 30.0, 30.0, 30.0


def calculate_lexical_diversity(text):
    """
    多维度词汇多样性计算
    返回：词型/词次比(TTR)、标准化TTR(MTLD)、词汇熵、稀有词占比
    """
    if not text.strip():
        return 0.0, 0.0, 0.0, 0.0
    
    # 分词（过滤标点和单字虚词）
    words = jieba.lcut(text)
    valid_words = [w for w in words if len(w) >= 2 and not re.match(r"[。，！？；：""''（）【】、]", w)]
    total_tokens = len(valid_words)  # 词次（Token count）
    if total_tokens < 5:  # 文本过短，返回兜底值
        return 0.3, 0.3, 1.0, 0.1
    
    # 1. 词型/词次比（Type-Token Ratio, TTR）
    unique_types = len(set(valid_words))  # 词型数（Type count）
    ttr = unique_types / total_tokens if total_tokens > 0 else 0.0
    ttr = clean_numeric_value(ttr, 0.3, 0.0, 1.0)
    
    # 2. 标准化TTR（MTLD）
    mtld_segments = []
    current_segment = []
    for word in valid_words:
        current_segment.append(word)
        current_unique = len(set(current_segment))
        current_ttr = current_unique / len(current_segment) if len(current_segment) > 0 else 0
        if current_ttr <= 0.72:  # 标准阈值
            mtld_segments.append(len(current_segment))
            current_segment = []
    # 处理剩余片段
    if current_segment:
        mtld_segments.append(len(current_segment))
    mtld = np.mean(mtld_segments) if mtld_segments else 0.0
    mtld_norm = mtld / 50  # 标准化到0-1范围
    mtld_norm = clean_numeric_value(mtld_norm, 0.3, 0.0, 1.0)
    
    # 3. 词汇熵（Shannon Entropy）
    word_count = Counter(valid_words)
    probs = [count / total_tokens for count in word_count.values()]
    probs = clean_array(probs, 0.0, 0.0, 1.0)
    entropy = -np.sum(p * np.log2(p + 1e-10) for p in probs)
    entropy_norm = entropy / np.log2(unique_types) if unique_types > 0 else 0.0
    entropy_norm = clean_numeric_value(entropy_norm, 0.5, 0.0, 1.0)
    
    # 4. 稀有词占比
    rare_word_count = sum(1 for count in word_count.values() if count <= 2)
    rare_word_ratio = rare_word_count / unique_types if unique_types > 0 else 0.0
    rare_word_ratio = clean_numeric_value(rare_word_ratio, 0.2, 0.0, 1.0)
    
    return round(ttr, 4), round(mtld_norm, 4), round(entropy_norm, 4), round(rare_word_ratio, 4)


def calculate_filler_repeat_freq(text, audio_duration):
    """
    计算填充词频率与重复词频率
    返回：总填充词频率、分类填充词频率、重复词频率、重复词占比
    """
    if not text.strip() or audio_duration <= 0:
        return 0.0, {k:0.0 for k in FILLER_WORDS.keys()}, 0.0, 0.0
    
    # 1. 填充词频率计算
    words = jieba.lcut(text)
    total_filler_count = 0
    filler_count_by_type = defaultdict(int)
    
    # 匹配填充词（优先长词）
    filler_sorted = sorted(ALL_FILLERS, key=lambda x: -len(x))
    text_temp = text
    for filler in filler_sorted:
        if filler in text_temp:
            count = text_temp.count(filler)
            total_filler_count += count
            # 确定填充词分类
            for filler_type, filler_list in FILLER_WORDS.items():
                if filler in filler_list:
                    filler_count_by_type[filler_type] += count
                    break
            text_temp = text_temp.replace(filler, "")
    
    # 填充词频率（个/分钟）
    filler_freq_total = (total_filler_count / audio_duration) * 60 if audio_duration > 0 else 0.0
    filler_freq_total = clean_numeric_value(filler_freq_total, 0.0, 0.0, 100.0)
    
    filler_freq_by_type = {
        filler_type: clean_numeric_value((count / audio_duration) * 60, 0.0, 0.0, 100.0) 
        for filler_type, count in filler_count_by_type.items()
    }
    # 补充未匹配到的分类
    for filler_type in FILLER_WORDS.keys():
        if filler_type not in filler_freq_by_type:
            filler_freq_by_type[filler_type] = 0.0
    
    # 2. 重复词频率计算
    # 连续重复词
    consecutive_repeat_count = 0
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            consecutive_repeat_count += 1
    
    # 非连续重复词
    valid_words = [w for w in words if len(w)>=2 and w not in ALL_FILLERS]
    word_count = Counter(valid_words)
    non_consecutive_repeat_count = sum(count - 2 for count in word_count.values() if count >=3)
    
    total_repeat_count = consecutive_repeat_count + non_consecutive_repeat_count
    repeat_freq = (total_repeat_count / audio_duration) * 60 if audio_duration > 0 else 0.0
    repeat_freq = clean_numeric_value(repeat_freq, 0.0, 0.0, 100.0)
    
    repeat_ratio = total_repeat_count / len(valid_words) if len(valid_words) > 0 else 0.0
    repeat_ratio = clean_numeric_value(repeat_ratio, 0.0, 0.0, 1.0)
    
    return (round(filler_freq_total, 4), 
            {k:round(v,4) for k,v in filler_freq_by_type.items()}, 
            round(repeat_freq, 4), 
            round(repeat_ratio, 4))


def calculate_language_complexity(text):
    if not text.strip():
        return 0.0, 0.0, 0.0
    
    words = jieba.lcut(text)
    if len(words) < 2:
        return 0.0, 0.0, 0.0
        
    word_counts = Counter(words)
    total_words = len(words)
    
    word_probs = [count/total_words for count in word_counts.values()]
    word_probs = clean_array(word_probs, 0.0, 0.0, 1.0)
    
    lexical_diversity = -np.sum(p * np.log2(p + 1e-10) for p in word_probs)
    lexical_diversity = clean_numeric_value(lexical_diversity, 3.0, 0.1, 10.0)
    
    avg_word_length = np.mean([len(word) for word in words])
    avg_word_length = clean_numeric_value(avg_word_length, 2.0, 1.0, 5.0)
    
    function_word_tags = {'uj', 'p', 'c', 'interj', 'y'}
    tagged_words = pseg.lcut(text)
    function_word_count = sum(1 for word, tag in tagged_words if tag in function_word_tags)
    syntactic_complexity = function_word_count / len(tagged_words) if len(tagged_words) > 0 else 0.0
    syntactic_complexity = clean_numeric_value(syntactic_complexity, 0.3, 0.0, 1.0)
    
    return round(lexical_diversity, 4), round(avg_word_length, 2), round(syntactic_complexity, 4)
