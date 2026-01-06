# -*- coding: utf-8 -*-
"""
embedding_server.py

一个更稳定的 SentenceTransformer Embedding 服务。

修复点：
1) 原文件重复定义了 /embed 路由，FastAPI 会以最后一个为准，行为不确定。
2) 原文件用全局 asyncio.Lock 把 encode 串行化，客户端并行请求也会排队。

策略：
- 用 Semaphore 控制最大并发（GPU 一般建议 1~2）
- 把阻塞的 encode 放到线程池（避免卡住 event loop）
- 提供 /embed 与 /embed_batch；客户端优先用 /embed_batch 提吞吐

环境变量（可选）：
- EMBED_DEVICE: cuda/cpu（默认 cuda）
- EMBED_BATCH_SIZE: encode batch_size（默认 64）
- EMBED_MAX_CONCURRENCY: 同时 encode 的最大并发（默认 1；GPU 通常 1~2）
- HOST / PORT: 启动监听（默认 127.0.0.1:8000）
"""

from __future__ import annotations

import os
import sys
import functools
from typing import List, Optional

import anyio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import torch

# 把项目根目录加入路径，以便读取 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GRAPHRAG_EMBEDDING_MODEL  # noqa: E402

app = FastAPI()

model_instance: Optional[SentenceTransformer] = None


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)


class EmbedBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)


def _get_env_int(key: str, default: int) -> int:
    try:
        v = int(str(os.getenv(key, str(default))).strip())
        return v
    except Exception:
        return default


def _clean_texts(texts: List[str]) -> List[str]:
    cleaned: List[str] = []
    for t in (texts or []):
        if t is None:
            t = ""
        s = str(t)
        if not s.strip():
            raise HTTPException(status_code=400, detail="Empty text in request")
        cleaned.append(s)
    if not cleaned:
        raise HTTPException(status_code=400, detail="No texts provided")
    return cleaned


def _encode_sync(texts: List[str], *, batch_size: int):
    """在同步线程中执行 encode（SentenceTransformer 是阻塞的）。"""
    if model_instance is None:
        raise RuntimeError("Model not loaded")
    return model_instance.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )



def _cleanup_cuda():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


async def _encode_with_retry(texts: List[str], *, batch_size: int):
    cur_bs = max(1, int(batch_size))
    while True:
        try:
            return await anyio.to_thread.run_sync(
                functools.partial(_encode_sync, texts, batch_size=cur_bs)
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" not in msg and "cuda" not in msg:
                raise
            _cleanup_cuda()
            if cur_bs <= 1:
                raise
            cur_bs = max(1, cur_bs // 2)
            print(f"[EmbeddingServer] OOM, retry with batch_size={cur_bs}")


@app.on_event("startup")
async def startup_event():
    global model_instance
    device = os.getenv("EMBED_DEVICE", "cuda")
    print(f"🚀 [EmbeddingServer] loading model={GRAPHRAG_EMBEDDING_MODEL} device={device}")
    model_instance = SentenceTransformer(GRAPHRAG_EMBEDDING_MODEL, device=device)
    print("✅ [EmbeddingServer] model ready")


# 并发控制：GPU 建议 1~2；CPU 可更大（看机器）
_MAX_CONCURRENCY = max(1, _get_env_int("EMBED_MAX_CONCURRENCY", 1))
_SEM = anyio.Semaphore(_MAX_CONCURRENCY)


@app.get("/health")
async def health():
    return {
        "ok": model_instance is not None,
        "model": GRAPHRAG_EMBEDDING_MODEL,
        "max_concurrency": _MAX_CONCURRENCY,
        "batch_size": _get_env_int("EMBED_BATCH_SIZE", 4),
    }


@app.post("/embed")
async def embed_text(req: EmbedRequest):
    if model_instance is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    batch_size = max(1, _get_env_int("EMBED_BATCH_SIZE", 4))
    texts = _clean_texts([req.text])

    async with _SEM:
        # encode 在后台线程执行，不阻塞 event loop
        embs = await _encode_with_retry(texts, batch_size=batch_size)

    return {"embedding": embs[0].tolist()}


@app.post("/embed_batch")
async def embed_batch(req: EmbedBatchRequest):
    if model_instance is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    batch_size = max(1, _get_env_int("EMBED_BATCH_SIZE", 4))
    texts = _clean_texts(req.texts)

    async with _SEM:
        embs = await _encode_with_retry(texts, batch_size=batch_size)

    return {"embeddings": [e.tolist() for e in embs]}


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = _get_env_int("PORT", 8000)
    # GPU 场景通常建议 workers=1，否则会重复加载模型占显存
    uvicorn.run(app, host=host, port=port)
