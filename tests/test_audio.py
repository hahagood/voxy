"""audio.py 测试"""

from unittest.mock import patch, MagicMock

import numpy as np


def test_list_devices():
    """测试设备列表功能。"""
    mock_devices = [
        {"name": "Test Mic", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Test Speaker", "max_input_channels": 0, "max_output_channels": 2},
    ]

    with patch("voxy.audio.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices
        mock_sd.default.device = (0, 1)

        from voxy.audio import list_devices
        result = list_devices()

        assert "Test Mic" in result
        assert "Test Speaker" not in result  # 输出设备不应列出
        assert "*" in result  # 默认设备标记


def test_list_devices_empty():
    """没有输入设备时的提示。"""
    with patch("voxy.audio.sd") as mock_sd:
        mock_sd.query_devices.return_value = [
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]
        mock_sd.default.device = (0, 0)

        from voxy.audio import list_devices
        result = list_devices()
        assert "未找到" in result
