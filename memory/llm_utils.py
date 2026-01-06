# -*- coding: utf-8 -*-
"""
LLM/JSON 兼容工具：
- 低温 bind 兼容不同 wrapper
- JSON dict 解析（json.loads + json_repair + 截取 {...}）
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_str(raw: str) -> str:
    s = (raw or "").strip()
    m = _JSON_FENCE_RE.search(s)
    if m:
        return m.group(1)

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def bind_llm_low_temp(llm):
    """
    兼容不同 LLM wrapper 的低温绑定写法。
    """
    try:
        return llm.bind(temperature=0.0)
    except Exception:
        pass
    try:
        return llm.bind_temperature(0.0)
    except Exception:
        pass
    return llm


def parse_json_dict(raw: str, llm=None) -> Dict[str, Any]:
    """
    尝试把 LLM 输出解析成 dict：
    1) json.loads
    2) json_repair.repair_json(return_objects=True)
    3) 截取第一段 {...} 再 json.loads
    失败则抛 ValueError
    """
    s = (raw or "").strip()
    json_str = _extract_json_str(s)

    # 1) strict json（优先对抽取后的 JSON 做）
    try:
        obj = json.loads(json_str)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) repair_json（同样对抽取后的 JSON 做）
    try:
        from json_repair import repair_json

        obj = repair_json(json_str, return_objects=True)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) extract {...}（再兜底一次：防止 extract 失败时仍有多余内容）
    start = json_str.find("{")
    end = json_str.rfind("}")
    if start != -1 and end != -1 and end > start:
        s2 = json_str[start : end + 1]
        try:
            obj2 = json.loads(s2)
            if isinstance(obj2, dict):
                return obj2
        except Exception:
            pass

    raise ValueError("Failed to parse JSON dict from LLM output.")
