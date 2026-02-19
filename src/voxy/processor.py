"""AI 文本处理模块"""

import os

from voxy.config import LLMConfig
from voxy.prompts import format_prompt


def _process_ollama(system_prompt: str, user_prompt: str, provider: str,
                    api_base: str, api_key: str = "", proxy: str = "") -> str:
    """直接调用 Ollama API，跳过 litellm 中间层。支持本地和云端。"""
    import httpx
    from urllib.parse import urlparse

    # provider 格式: "ollama/model_name"
    model = provider.split("/", 1)[1]

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # 本地地址不走代理；云端地址使用配置的代理或环境变量
    host = urlparse(api_base).hostname or ""
    is_local = host in ("localhost", "127.0.0.1", "::1")

    # 云端模型 (如 :cloud) 响应较慢，需要更长超时
    timeout = 180.0 if ":cloud" in model else 60.0
    client_kwargs: dict = {"timeout": timeout}
    if is_local:
        client_kwargs["trust_env"] = False
    elif proxy:
        client_kwargs["proxy"] = proxy

    with httpx.Client(**client_kwargs) as client:
        resp = client.post(
            f"{api_base}/api/chat",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.3},
            },
        )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _process_litellm(system_prompt: str, user_prompt: str, provider: str,
                     api_base: str, api_key: str, proxy: str) -> str:
    """通过 litellm 调用（云端模型等非 ollama 场景）。"""
    import logging

    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

    import litellm

    litellm.suppress_debug_info = True
    litellm.drop_params = True
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
    logging.getLogger("litellm").setLevel(logging.CRITICAL)

    kwargs = {}
    if api_base and "localhost" not in api_base and "127.0.0.1" not in api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    # 设置代理（SOCKS5 等）
    old_proxy = os.environ.get("HTTPS_PROXY")
    old_all_proxy = os.environ.get("ALL_PROXY")
    if proxy:
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["ALL_PROXY"] = proxy

    try:
        response = litellm.completion(
            model=provider,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
            **kwargs,
        )
    finally:
        if proxy:
            if old_proxy is None:
                os.environ.pop("HTTPS_PROXY", None)
            else:
                os.environ["HTTPS_PROXY"] = old_proxy
            if old_all_proxy is None:
                os.environ.pop("ALL_PROXY", None)
            else:
                os.environ["ALL_PROXY"] = old_all_proxy

    return response.choices[0].message.content


def _call_llm(system_prompt: str, user_prompt: str, provider: str,
              api_base: str, api_key: str, proxy: str) -> str:
    """根据 provider 类型选择调用方式。"""
    if provider.startswith("ollama/"):
        return _process_ollama(system_prompt, user_prompt, provider, api_base, api_key, proxy)
    else:
        return _process_litellm(system_prompt, user_prompt, provider, api_base, api_key, proxy)


def process_text(raw_text: str, config: LLMConfig) -> str:
    """用 LLM 润色语音转写文本。短文本用本地模型，长文本用大模型。"""
    if not raw_text.strip():
        return raw_text

    system_prompt, user_prompt = format_prompt(raw_text, config.custom_terms)

    # 长文本且配置了大模型 → 切换到大模型
    use_long = (config.long_provider
                and len(raw_text) > config.long_threshold)

    if use_long:
        provider = config.long_provider
        api_base = config.long_api_base
        api_key = config.long_api_key
        proxy = config.long_proxy
    else:
        provider = config.provider
        api_base = config.api_base
        api_key = config.api_key
        proxy = config.proxy

    result = _call_llm(system_prompt, user_prompt, provider, api_base, api_key, proxy)

    return result.strip() if result else raw_text
