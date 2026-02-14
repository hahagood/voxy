"""AI 文本处理模块 - litellm 集成"""

from voxy.config import LLMConfig
from voxy.prompts import format_prompt


def process_text(raw_text: str, config: LLMConfig) -> str:
    """用 LLM 润色语音转写文本。

    Args:
        raw_text: 原始转写文本
        config: LLM 配置

    Returns:
        润色后的文本
    """
    if not raw_text.strip():
        return raw_text

    import logging

    import litellm

    # 静音 litellm 日志
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
    logging.getLogger("litellm").setLevel(logging.CRITICAL)

    system_prompt, user_prompt = format_prompt(raw_text)

    kwargs = {}
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.api_key:
        kwargs["api_key"] = config.api_key

    response = litellm.completion(
        model=config.provider,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=512,
        **kwargs,
    )

    result = response.choices[0].message.content
    return result.strip() if result else raw_text
