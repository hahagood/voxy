"""云端 OpenAI Whisper API STT 后端"""

import io
import os
import wave

import numpy as np

from voxy.config import STTConfig
from voxy.stt import STTEngine


class CloudSTT(STTEngine):
    """基于 OpenAI Whisper API 的云端语音识别。"""

    def __init__(self, config: STTConfig):
        self._config = config

    def _audio_to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> io.BytesIO:
        """将 float32 numpy array 编码为 WAV 格式的 BytesIO。"""
        # 转换为 int16 PCM
        pcm = (audio * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

        buf.seek(0)
        buf.name = "audio.wav"
        return buf

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        from openai import OpenAI

        cc = self._config.cloud
        api_key = cc.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("云端 STT 需要设置 api_key 或 OPENAI_API_KEY 环境变量")

        client = OpenAI(api_key=api_key, base_url=cc.api_base)

        wav_buf = self._audio_to_wav_bytes(audio, sample_rate)

        language = self._config.language
        if language == "auto":
            language = None

        kwargs = {"model": cc.model, "file": wav_buf}
        if language:
            kwargs["language"] = language

        response = client.audio.transcriptions.create(**kwargs)
        return response.text.strip()
