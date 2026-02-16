"""本地 SenseVoice STT 后端 (funasr)"""

import gc
import re

import numpy as np

from voxy.config import STTConfig
from voxy.stt import STTEngine


def rich_transcription_postprocess(text: str) -> str:
    """SenseVoice 后处理：移除特殊标记。"""
    # 移除 <|xxx|> 格式的标记 (如 <|zh|>, <|NEUTRAL|>, <|Speech|> 等)
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return text.strip()


class SenseVoiceSTT(STTEngine):
    """基于 FunASR SenseVoice 的本地语音识别。"""

    def __init__(self, config: STTConfig):
        self._config = config
        self._model = None

    def _load_model(self):
        """懒加载模型。"""
        if self._model is not None:
            return

        import io
        import logging
        import sys
        import warnings

        # 静音所有 funasr/modelscope 噪音
        logging.getLogger("modelscope").setLevel(logging.CRITICAL)
        logging.getLogger("funasr").setLevel(logging.CRITICAL)
        logging.disable(logging.WARNING)
        warnings.filterwarnings("ignore")

        sc = self._config.sensevoice
        print(f"  加载 SenseVoice 模型: {sc.model} (设备: {sc.device})", file=sys.stderr, flush=True)

        # 静音 import 和 AutoModel 初始化时的 stdout/stderr 噪音
        stderr_backup = sys.stderr
        stdout_backup = sys.stdout
        sys.stderr = io.StringIO()
        sys.stdout = io.StringIO()
        try:
            from funasr import AutoModel

            self._model = AutoModel(
                model=sc.model,
                trust_remote_code=True,
                device=sc.device,
                disable_update=True,
            )
        finally:
            sys.stdout = stdout_backup
            sys.stderr = stderr_backup
            logging.disable(logging.NOTSET)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        self._load_model()

        language = self._config.language
        if language == "auto":
            language = "auto"

        import io
        import sys

        # 静音 funasr generate 的 tqdm 和 rtf_avg 输出
        stderr_backup = sys.stderr
        sys.stderr = io.StringIO()
        try:
            result = self._model.generate(
                input=audio,
                cache={},
                language=language,
                use_itn=True,
                batch_size_s=0,
            )
        finally:
            sys.stderr = stderr_backup

        if not result:
            return ""

        text = result[0].get("text", "")
        text = rich_transcription_postprocess(text)
        return text

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
