"""Lazy Dolphin speech recognition; no model is loaded on import."""
import os
import re
import torch
import dolphin
from zhconv import convert
_preferred_dolphin_model = os.environ.get("MPCAS_AUDIO_DOLPHIN_MODEL", "small")
if _preferred_dolphin_model != "small":
    raise RuntimeError("MPCAS_AUDIO_DOLPHIN_MODEL must be small")
asr_model = None


def safe_set_gpu(device_id=1):  # 改为设备ID=1（你的日志显示使用的是ID=1的RTX 4090）
    if torch.cuda.is_available():
        if device_id < torch.cuda.device_count():
            torch.cuda.set_device(device_id)
            print(f"✅ 使用GPU: {torch.cuda.get_device_name(device_id)} (设备ID: {device_id})")
        else:
            default_device = torch.cuda.current_device()
            print(f"⚠️ 指定GPU设备ID {device_id}不存在，自动使用默认GPU: {torch.cuda.get_device_name(default_device)}")
    else:
        print("⚠️ 未检测到GPU，使用CPU（处理速度会较慢）")


def safe_load_dolphin_model(preferred_model="small"):
    """
    加载 Dolphin ASR 模型：
    - 只允许 small，加载失败时绝不回退到 base
    - 支持 GPU / CPU 自动切换，但两者使用相同的 small 权重
    """
    if preferred_model != "small":
        raise ValueError(f"本流程强制使用 Dolphin small，拒绝加载：{preferred_model!r}")

    # small 使用独立目录，防止其他模型下载时覆盖共用的 train.yaml。
    model_dir = os.environ.get(
        "MPCAS_AUDIO_DOLPHIN_MODEL_DIR",
        os.path.expanduser("~/.cache/mpcas/dolphin-small"),
    )
    os.makedirs(model_dir, exist_ok=True)

    requested_device = os.environ.get("MPCAS_AUDIO_DOLPHIN_DEVICE", "auto").strip().lower()
    if requested_device == "cpu":
        candidate_devices = ["cpu"]
    elif requested_device == "cuda":
        candidate_devices = ["cuda", "cpu"]
    else:
        candidate_devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    print(f"??? Dolphin ??????{','.join(candidate_devices)}")

    last_error = None
    for device in candidate_devices:
        try:
            print(f"正在加载 Dolphin small（dir={model_dir}, device={device}）...")
            model = dolphin.load_model("small", model_dir, device)
            print(f"Dolphin small 加载成功（device={device}）")
            return model
        except Exception as e:
            last_error = e
            print(f"Dolphin small 在 {device} 加载失败：{str(e)[:120]}")
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    raise RuntimeError(f"Dolphin small 加载失败；已禁止回退到 base。最后错误：{last_error}")


def transcribe_with_dolphin(audio_path, prompt_text=""):
    """
    使用 Dolphin 进行语音识别：
    - 仍然使用任务提示词 prompt_text（只用于日志，不直接喂给模型）
    - 指定中文 zh + 中国地区 CN
    - 保留原来的“去除连续重复短语”逻辑
    """
    try:
        decoding_method = os.environ.get("MPCAS_AUDIO_DOLPHIN_DECODING", "attention_rescoring")
        beam_size = int(os.environ.get("MPCAS_AUDIO_DOLPHIN_BEAM_SIZE", "10"))

        # dataoceanai-dolphin 20260513 exposes lang/region on the high-level
        # transcribe API, not on ASRModel.forward.
        global asr_model
        if asr_model is None:
            asr_model = safe_load_dolphin_model(_preferred_dolphin_model)
        try:
            result = dolphin.transcribe(
                asr_model,
                audio_path,
                lang_sym="zh",
                region_sym="CN",
                decoding_method=decoding_method,
                beam_size=beam_size,
            )
        except Exception as transcribe_error:
            message = str(transcribe_error)
            if "CUDA out of memory" not in message and "cuda" not in message.lower():
                raise
            print(f"?? Dolphin GPU????????????CPU??: {message[:160]}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            previous_device = os.environ.get("MPCAS_AUDIO_DOLPHIN_DEVICE")
            os.environ["MPCAS_AUDIO_DOLPHIN_DEVICE"] = "cpu"
            asr_model = safe_load_dolphin_model(_preferred_dolphin_model)
            if previous_device is not None:
                os.environ["MPCAS_AUDIO_DOLPHIN_DEVICE"] = previous_device
            else:
                os.environ.pop("MPCAS_AUDIO_DOLPHIN_DEVICE", None)
            result = dolphin.transcribe(
                asr_model,
                audio_path,
                lang_sym="zh",
                region_sym="CN",
                decoding_method=decoding_method,
                beam_size=beam_size,
            )

        # Dolphin 的结果对象一般有 .text 属性
        if hasattr(result, "text"):
            raw_text = (result.text or "").strip()
        elif isinstance(result, dict):
            raw_text = (result.get("text") or "").strip()
        else:
            raw_text = str(result).strip()

        simplified_text = convert(raw_text, 'zh-cn') if raw_text else "未识别到有效文本"
        # Current Dolphin releases may prefix decoded text with control tokens
        # such as <zh><CN><asr><notimestamp>.  They are model metadata, not
        # spoken content, and would otherwise contaminate lexical features.
        simplified_text = re.sub(r"^(?:\s*<[^>]+>)+\s*", "", simplified_text)

        # 去除连续重复短语（保留你原来的去重规则）
        simplified_text = re.sub(r"([\u4e00-\u9fa5]{4,})(\1+)", r"\1", simplified_text)

        print(f"🗣️ Dolphin识别结果（去重后）: {simplified_text}")
        if prompt_text:
            print(f"📌 任务提示词（仅用于分析，不直接输入模型）: {prompt_text}")

        return simplified_text

    except Exception as e:
        # 这里不再回退到 Whisper，而是直接返回失败提示
        print(f"❌ Dolphin 语音识别失败: {str(e)}")
        return "语音识别完全失败"
