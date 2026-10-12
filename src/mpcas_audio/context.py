"""Task prompts and record-path parsing for speech features."""
import os
import re
DATASET_DIR = os.environ.get("MPCAS_AUDIO_DATASET_DIR", "audio_wav")
WORD_RECALL_TARGETS = {
    "T004_PRACTICE": ["苹果", "桌子", "火车"],
    "T004_FORMAL": ["大衣", "司机", "海棠", "木工", "长裤", "百合", "头巾", "腊梅", "士兵", "玉兰", "律师", "手套"],
    "T009_FORMAL": ["大衣", "司机", "海棠", "木工", "长裤", "百合", "头巾", "腊梅", "士兵", "玉兰", "律师", "手套"],
}
TASK_PROMPTS = {task: "，".join(words) for task, words in WORD_RECALL_TARGETS.items()}
STORY_TASKS = {"T013_FORMAL", "T013_PRACTICE"}


def infer_task_context(audio_path):
    """
    根据文件路径推断任务类型（词语回忆/图片讲述），返回提示词和目标词
    """
    parts = [p.lower() for p in os.path.normpath(audio_path).split(os.sep)]
    task_id = next((p.upper() for p in parts if re.match(r"t0\d{2}", p, re.IGNORECASE)), "")
    stage = "FORMAL" if "formal" in parts else "PRACTICE" if "practice" in parts else ""
    task_key = f"{task_id}_{stage}" if task_id and stage else task_id

    target_words = WORD_RECALL_TARGETS.get(task_key, [])
    prompt_text = TASK_PROMPTS.get(task_key, "")
    task_type = "recall" if target_words else ("story" if task_key in STORY_TASKS else "unknown")

    return {
        "task_key": task_key,
        "task_type": task_type,
        "target_words": target_words,
        "prompt_text": prompt_text
    }


def build_subpath_index(audio_path):
    """
    Generate subpath index like E0103_T004 from dataset-relative path.
    Falls back to filename stem if the expected structure is missing.
    """
    try:
        rel_path = os.path.relpath(audio_path, DATASET_DIR)
        parts = os.path.normpath(rel_path).split(os.sep)
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    except Exception:
        pass
    return os.path.splitext(os.path.basename(audio_path))[0]


def extract_path_fields(audio_path):
    """
    Split dataset-relative path into subject / task / phase (e.g., E0102 / T004 / formal).
    Falls back to empty strings if the structure is missing.
    """
    try:
        rel_path = os.path.relpath(audio_path, DATASET_DIR)
        parts = os.path.normpath(rel_path).split(os.sep)
        if parts and os.path.splitext(parts[-1])[1]:
            parts = parts[:-1]
        subject = parts[0] if len(parts) > 0 else ""
        task = parts[1] if len(parts) > 1 else ""
        phase = parts[2] if len(parts) > 2 else ""
        phase = phase.lower() if phase else ""
        return subject, task, phase
    except Exception:
        return "", "", ""
