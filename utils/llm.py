# utils/llm.py
from __future__ import annotations

from typing import Any, Dict, Optional
from langchain_openai import ChatOpenAI
from config import (
    GRAPHRAG_API_BASE, GRAPHRAG_CHAT_API_KEY, GRAPHRAG_CHAT_MODEL,
    CHEAP_GRAPHRAG_CHAT_MODEL, CHEAP_GRAPHRAG_API_BASE, CHEAP_GRAPHRAG_CHAT_API_KEY
)

# =========================
# Global singletons
# =========================
_CHEAP_LLM: Optional[ChatOpenAI] = None
_EXPENSIVE_LLM: Optional[ChatOpenAI] = None

def _make_llm(*, model: str, api_key: str, base_url: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        request_timeout=900,
        max_retries=3,
    )

def get_expensive_llm() -> ChatOpenAI:
    """
    获取并返回一个配置好的大语言模型实例（昂贵模型，单例）。
    """
    global _EXPENSIVE_LLM
    if _EXPENSIVE_LLM is not None:
        return _EXPENSIVE_LLM

    if not all([GRAPHRAG_API_BASE, GRAPHRAG_CHAT_API_KEY, GRAPHRAG_CHAT_MODEL]):
        raise ValueError("LLM 配置不完整，无法初始化模型。请检查 .env 文件。")

    _EXPENSIVE_LLM = _make_llm(
        model=GRAPHRAG_CHAT_MODEL,
        api_key=GRAPHRAG_CHAT_API_KEY,
        base_url=GRAPHRAG_API_BASE,
        temperature=0.7,
    )
    return _EXPENSIVE_LLM

def get_llm() -> ChatOpenAI:
    """
    获取并返回一个配置好的大语言模型实例（便宜模型，单例）。
    """
    global _CHEAP_LLM
    if _CHEAP_LLM is not None:
        return _CHEAP_LLM

    if not all([CHEAP_GRAPHRAG_CHAT_MODEL, CHEAP_GRAPHRAG_API_BASE, CHEAP_GRAPHRAG_CHAT_API_KEY]):
        raise ValueError("LLM 配置不完整，无法初始化模型。请检查 .env 文件。")

    _CHEAP_LLM = _make_llm(
        model=CHEAP_GRAPHRAG_CHAT_MODEL,
        api_key=CHEAP_GRAPHRAG_CHAT_API_KEY,
        base_url=CHEAP_GRAPHRAG_API_BASE,
        temperature=0.7,
    )
    return _CHEAP_LLM

def get_llm_low_temp() -> ChatOpenAI:
    """便宜模型（低温版），复用同一个 client，通过 bind 覆盖温度。"""
    return get_llm().bind(temperature=0.0)

def get_expensive_llm_low_temp() -> ChatOpenAI:
    """昂贵模型（低温版）。"""
    return get_expensive_llm().bind(temperature=0.0)

def invoke_json(llm: ChatOpenAI, prompt: str) -> str:
    """
    尽量强制模型只输出 JSON。
    - 优先使用 response_format（不同供应商/版本可能支持情况不同）
    - 不支持则退化为普通 invoke
    返回：模型输出的 raw string（JSON 文本，可能带```，上层可 repair_json）
    """
    # 尝试强制 JSON 输出
    try:
        llm_json = llm.bind(response_format={"type": "json_object"})
        return llm_json.invoke(prompt).content.strip()
    except Exception:
        # fallback：普通输出
        return llm.invoke(prompt).content.strip()
