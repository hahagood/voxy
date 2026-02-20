"""语音命令匹配模块"""

import json
import re
import subprocess
from difflib import SequenceMatcher

# STT 常添加的尾部标点
_TRAILING_PUNCT = re.compile(r'[。，！？、；：.!?,;:\s]+$')


def get_active_window_class() -> str | None:
    """获取 Hyprland 当前焦点窗口的 class（小写）。"""
    try:
        result = subprocess.run(
            ["hyprctl", "activewindow", "-j"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            cls = data.get("class", "")
            return cls.lower() if cls else None
    except Exception:
        pass
    return None


def _resolve_map(
    command_map: dict[str, str],
    context: dict[str, dict[str, str]],
    window_class: str | None,
) -> dict[str, str]:
    """合并 context 覆盖到 command_map，返回最终映射。"""
    if not window_class or not context:
        return command_map

    for pattern, overrides in context.items():
        if pattern.lower() in window_class:
            merged = dict(command_map)
            merged.update(overrides)
            return merged

    return command_map


def match_command(
    text: str,
    command_map: dict[str, str],
    fuzzy_threshold: float = 0.0,
    *,
    window_class: str | None = None,
    context: dict[str, dict[str, str]] | None = None,
) -> tuple[str, str] | None:
    """匹配转写文本到命令。

    Args:
        text: 转写后的文本（已 strip）
        command_map: {触发词: 动作字符串} 映射
        fuzzy_threshold: 模糊匹配阈值 (0-1)，0 表示仅精确匹配
        window_class: 当前焦点窗口 class（小写）
        context: {窗口模式: {触发词: 动作}} 上下文覆盖映射

    Returns:
        (触发词, 动作字符串) 或 None
    """
    if not command_map or not text.strip():
        return None

    normalized = _TRAILING_PUNCT.sub('', text.strip())
    if not normalized:
        return None

    # 合并 context 覆盖
    effective_map = _resolve_map(command_map, context or {}, window_class)

    # 精确匹配优先
    if normalized in effective_map:
        return (normalized, effective_map[normalized])

    # 模糊匹配（仅当 threshold > 0）
    if fuzzy_threshold > 0:
        best_match = None
        best_ratio = 0.0
        for trigger, action in effective_map.items():
            ratio = SequenceMatcher(None, normalized, trigger).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = (trigger, action)
        if best_match and best_ratio >= fuzzy_threshold:
            return best_match

    return None
