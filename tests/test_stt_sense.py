"""SenseVoice STT 后端测试"""

import sys
from unittest.mock import patch, MagicMock

import numpy as np

from voxy.config import STTConfig, SenseVoiceConfig
from voxy.stt.local_sense import rich_transcription_postprocess


def test_rich_transcription_postprocess():
    """测试 SenseVoice 后处理。"""
    text = "<|zh|><|NEUTRAL|><|Speech|>你好世界"
    assert rich_transcription_postprocess(text) == "你好世界"


def test_postprocess_empty():
    assert rich_transcription_postprocess("") == ""


def test_postprocess_no_tags():
    assert rich_transcription_postprocess("纯文本") == "纯文本"


def test_sensevoice_transcribe():
    """测试 SenseVoice 转写功能。"""
    config = STTConfig(
        backend="sensevoice",
        language="zh",
        sensevoice=SenseVoiceConfig(model="iic/SenseVoiceSmall", device="cpu"),
    )

    mock_funasr = MagicMock()
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"text": "<|zh|><|NEUTRAL|><|Speech|>你好世界"}]
    mock_funasr.AutoModel.return_value = mock_model

    with patch.dict(sys.modules, {"funasr": mock_funasr}):
        if "voxy.stt.local_sense" in sys.modules:
            del sys.modules["voxy.stt.local_sense"]

        from voxy.stt.local_sense import SenseVoiceSTT
        engine = SenseVoiceSTT(config)
        result = engine.transcribe(np.zeros(16000, dtype=np.float32))

        assert result == "你好世界"
