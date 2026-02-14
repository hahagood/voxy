"""processor.py 测试"""

from unittest.mock import patch, MagicMock

from voxy.config import LLMConfig


def test_process_text():
    """测试 LLM 润色功能。"""
    config = LLMConfig(
        provider="ollama/qwen2.5:7b",
        api_base="http://localhost:11434",
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "你好，世界。"

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        from voxy.processor import process_text
        result = process_text("嗯那个你好啊世界", config)

        assert result == "你好，世界。"
        mock_completion.assert_called_once()


def test_process_empty_text():
    """空文本直接返回。"""
    config = LLMConfig()

    from voxy.processor import process_text
    result = process_text("", config)
    assert result == ""


def test_process_whitespace_text():
    """纯空白文本直接返回。"""
    config = LLMConfig()

    from voxy.processor import process_text
    result = process_text("   ", config)
    assert result == "   "
