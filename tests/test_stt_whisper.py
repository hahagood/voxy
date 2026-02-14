"""faster-whisper STT 后端测试"""

import sys
from unittest.mock import patch, MagicMock, PropertyMock
from types import ModuleType

import numpy as np

from voxy.config import STTConfig, WhisperConfig


def _make_mock_faster_whisper():
    """创建 faster_whisper 模块 mock。"""
    mock_module = MagicMock()
    return mock_module


def test_whisper_transcribe():
    """测试 Whisper 转写功能。"""
    config = STTConfig(
        backend="whisper",
        language="zh",
        whisper=WhisperConfig(model="tiny", device="cpu", compute_type="int8"),
    )

    mock_segment = MagicMock()
    mock_segment.text = "你好世界"
    mock_info = MagicMock()

    mock_fw = _make_mock_faster_whisper()
    mock_fw.WhisperModel.return_value.transcribe.return_value = ([mock_segment], mock_info)

    with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
        # Force reimport
        if "voxy.stt.local_whisper" in sys.modules:
            del sys.modules["voxy.stt.local_whisper"]

        from voxy.stt.local_whisper import WhisperSTT
        engine = WhisperSTT(config)
        result = engine.transcribe(np.zeros(16000, dtype=np.float32))

        assert result == "你好世界"
        mock_fw.WhisperModel.assert_called_once_with("tiny", device="cpu", compute_type="int8")


def test_whisper_auto_language():
    """language=auto 时传 None 给 Whisper。"""
    config = STTConfig(
        backend="whisper",
        language="auto",
        whisper=WhisperConfig(model="tiny", device="cpu", compute_type="int8"),
    )

    mock_fw = _make_mock_faster_whisper()
    mock_fw.WhisperModel.return_value.transcribe.return_value = ([], MagicMock())

    with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
        if "voxy.stt.local_whisper" in sys.modules:
            del sys.modules["voxy.stt.local_whisper"]

        from voxy.stt.local_whisper import WhisperSTT
        engine = WhisperSTT(config)
        engine.transcribe(np.zeros(16000, dtype=np.float32))

        call_kwargs = mock_fw.WhisperModel.return_value.transcribe.call_args
        assert call_kwargs[1]["language"] is None
