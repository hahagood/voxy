"""STT 引擎基类 + 工厂函数"""

from abc import ABC, abstractmethod

import numpy as np

from voxy.config import STTConfig


class STTEngine(ABC):
    """语音识别引擎抽象基类。"""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """将音频转为文字。

        Args:
            audio: float32 numpy array, 单声道
            sample_rate: 采样率 (Hz)

        Returns:
            识别出的文字
        """
        ...


def create_stt(config: STTConfig) -> STTEngine:
    """根据配置创建 STT 引擎。"""
    backend = config.backend.lower()

    if backend == "whisper":
        from voxy.stt.local_whisper import WhisperSTT
        return WhisperSTT(config)
    elif backend == "sensevoice":
        from voxy.stt.local_sense import SenseVoiceSTT
        return SenseVoiceSTT(config)
    elif backend == "cloud":
        from voxy.stt.cloud import CloudSTT
        return CloudSTT(config)
    else:
        raise ValueError(f"未知的 STT 后端: {backend}，支持: whisper / sensevoice / cloud")
