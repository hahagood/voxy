"""语音命令匹配模块"""

import re
from difflib import SequenceMatcher

# STT 常添加的尾部标点
_TRAILING_PUNCT = re.compile(r'[。，！？、；：.!?,;:\s]+$')


def match_command(
    text: str,
    command_map: dict[str, str],
    fuzzy_threshold: float = 0.0,
) -> tuple[str, str] | None:
    """匹配转写文本到命令。

    Args:
        text: 转写后的文本（已 strip）
        command_map: {触发词: 动作字符串} 映射
        fuzzy_threshold: 模糊匹配阈值 (0-1)，0 表示仅精确匹配

    Returns:
        (触发词, 动作字符串) 或 None
    """
    if not command_map or not text.strip():
        return None

    normalized = _TRAILING_PUNCT.sub('', text.strip())
    if not normalized:
        return None

    # 精确匹配优先
    if normalized in command_map:
        return (normalized, command_map[normalized])

    # 模糊匹配（仅当 threshold > 0）
    if fuzzy_threshold > 0:
        best_match = None
        best_ratio = 0.0
        for trigger, action in command_map.items():
            ratio = SequenceMatcher(None, normalized, trigger).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = (trigger, action)
        if best_match and best_ratio >= fuzzy_threshold:
            return best_match

    return None
