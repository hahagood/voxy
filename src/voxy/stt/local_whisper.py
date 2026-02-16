"""本地 faster-whisper STT 后端"""

import gc
import sys

import numpy as np

from voxy.config import STTConfig
from voxy.stt import STTEngine


class WhisperSTT(STTEngine):
    """基于 faster-whisper 的本地语音识别。"""

    def __init__(self, config: STTConfig):
        self._config = config
        self._model = None

    def _load_model(self):
        """懒加载模型。"""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        wc = self._config.whisper
        print(f"  加载 Whisper 模型: {wc.model} (设备: {wc.device}, 精度: {wc.compute_type})", file=sys.stderr)
        self._model = WhisperModel(
            wc.model,
            device=wc.device,
            compute_type=wc.compute_type,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        self._load_model()

        language = self._config.language
        if language == "auto":
            language = None

        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        text = "".join(seg.text for seg in segments).strip()
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
