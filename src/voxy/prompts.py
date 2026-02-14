"""LLM 提示词模板"""

SYSTEM_PROMPT = """\
你是语音转文字后处理助手。直接输出润色后的文本，不要解释。
规则：去语气词、去重复、修正错别字、加标点。保持原意，不添加内容。/no_think\
"""

USER_PROMPT_TEMPLATE = "{text}"


def format_prompt(raw_text: str) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt) 元组。"""
    return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(text=raw_text)
