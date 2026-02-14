"""云端 STT 后端测试"""

import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np

from voxy.config import STTConfig, CloudSTTConfig


def test_cloud_audio_to_wav():
    """测试 audio → WAV 编码。"""
    config = STTConfig(
        backend="cloud",
        cloud=CloudSTTConfig(api_key="test-key"),
    )

    from voxy.stt.cloud import CloudSTT
    engine = CloudSTT(config)

    audio = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
    wav_buf = engine._audio_to_wav_bytes(audio, 16000)

    # 验证 WAV header
    data = wav_buf.read(4)
    assert data == b"RIFF"


def test_cloud_transcribe():
    """测试云端转写功能。"""
    config = STTConfig(
        backend="cloud",
        language="zh",
        cloud=CloudSTTConfig(api_key="test-key", model="whisper-1"),
    )

    mock_response = MagicMock()
    mock_response.text = "你好世界"

    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_response
    mock_openai.OpenAI.return_value = mock_client

    with patch.dict(sys.modules, {"openai": mock_openai}):
        if "voxy.stt.cloud" in sys.modules:
            del sys.modules["voxy.stt.cloud"]

        from voxy.stt.cloud import CloudSTT
        engine = CloudSTT(config)
        result = engine.transcribe(np.zeros(16000, dtype=np.float32))

        assert result == "你好世界"


def test_cloud_missing_api_key():
    """没有 API key 时应报错。"""
    config = STTConfig(
        backend="cloud",
        cloud=CloudSTTConfig(api_key=""),
    )

    from voxy.stt.cloud import CloudSTT
    engine = CloudSTT(config)

    # 确保没有环境变量
    with patch.dict(os.environ, {}, clear=True):
        try:
            engine.transcribe(np.zeros(16000, dtype=np.float32))
            assert False, "应该抛出 ValueError"
        except ValueError as e:
            assert "api_key" in str(e)
