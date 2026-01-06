# utils/token_chunker.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List


def _get_tiktoken_encoding(encoding_model: str):
    """
    encoding_model can be:
      - an encoding name (e.g., "cl100k_base")
      - a model name (e.g., "gpt-4o-mini"), in which case we try encoding_for_model
    """
    import tiktoken

    name = (encoding_model or "").strip()
    if not name:
        name = "cl100k_base"

    # 1) Try treat as encoding name
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        pass

    # 2) Try treat as model name
    try:
        return tiktoken.encoding_for_model(name)
    except Exception:
        # 3) Safe fallback
        return tiktoken.get_encoding("cl100k_base")


def chunk_text_by_tokens(
    text: str,
    *,
    chunk_size: int = 1024,
    overlap: int = 512,
    encoding_model: str = "cl100k_base",
) -> List[str]:
    """
    GraphRAG-like token sliding window chunking.

    - chunk_size <= 0: return [full_text]
    - overlap is auto-clamped into [0, chunk_size-1]
    - encoding_model supports both encoding name and model name
    """
    s = (text or "").strip()
    if not s:
        return []

    if chunk_size <= 0:
        return [s]

    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    try:
        enc = _get_tiktoken_encoding(encoding_model)
    except Exception as e:
        raise RuntimeError("缺少依赖 tiktoken：请 pip install tiktoken") from e

    ids = enc.encode(s)
    if not ids:
        return [s]

    step = chunk_size - overlap
    if step <= 0:
        step = 1

    out: List[str] = []
    last = None

    for start in range(0, len(ids), step):
        window = ids[start : start + chunk_size]
        if not window:
            break
        chunk = enc.decode(window).strip()
        if not chunk:
            continue

        # Avoid immediate duplicates (common when text is short or overlap is huge)
        if last is not None and chunk == last:
            continue
        last = chunk
        out.append(chunk)

        # If we've reached the end, stop (avoid generating many tiny tail windows)
        if start + chunk_size >= len(ids):
            break

    # Global dedup (keeps order)
    dedup: List[str] = []
    seen = set()
    for c in out:
        if c in seen:
            continue
        seen.add(c)
        dedup.append(c)

    return dedup
