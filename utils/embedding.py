import os
import requests
import numpy as np
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from config import (
    GRAPHRAG_EMBEDDING_API_KEY,
    GRAPHRAG_EMBEDDING_MODEL,
    GRAPHRAG_EMBEDDING_API_BASE
)

_GLOBAL_EMBEDDING_MODEL = None

# ----------------------------
# 模式判断逻辑
# ----------------------------
BASE = str(GRAPHRAG_EMBEDDING_API_BASE or "").strip().rstrip("/")

# 1) 远程服务模式：当 API_BASE 指向 localhost/127.0.0.1（你现在用这个）
USE_REMOTE_SERVER = False
if BASE and ("127.0.0.1" in BASE or "localhost" in BASE):
    USE_REMOTE_SERVER = True

# 2) 本地模型模式：API_BASE 为空 或 API_KEY=local
USE_LOCAL_MODEL = False
if (not BASE) or ("local" in str(GRAPHRAG_EMBEDDING_API_KEY or "").lower()):
    USE_LOCAL_MODEL = True


# =================================================================
# 1. 远程服务模式（推荐）
# =================================================================
class RemoteEmbeddingClient(Embeddings):
    """
    连接后台 Embedding Server 的轻量级客户端
    约定：
      - POST {base_url}/embed       {"text": "..."} -> {"embedding": [...]}
      - POST {base_url}/embed_batch {"texts":[...]}  -> {"embeddings":[[...], ...]}
    """
    def __init__(self, base_url: str, timeout_s: int = 60):
        self.base_url = (base_url or "").strip().rstrip("/")
        self.timeout_s = int(timeout_s)

        self._single_url = f"{self.base_url}/embed"
        self._batch_url = f"{self.base_url}/embed_batch"

        # 是否可用 batch：首次探测后缓存
        self._batch_available: Optional[bool] = None

    def _check_batch_available(self) -> bool:
        """
        探测 /embed_batch 是否存在。
        - 不额外依赖 /health，直接用一次空 batch 轻量探测
        """
        if self._batch_available is not None:
            return self._batch_available

        try:
            resp = requests.post(
                self._batch_url,
                json={"texts": ["ping"]},
                timeout=min(self.timeout_s, 10),
            )
            self._batch_available = (resp.status_code == 200)
        except Exception:
            self._batch_available = False

        return self._batch_available

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # 优先走 batch
        if self._check_batch_available():
            try:
                resp = requests.post(
                    self._batch_url,
                    json={"texts": texts},
                    timeout=self.timeout_s,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embs = data.get("embeddings", [])
                    # 简单校验：长度要对得上
                    if isinstance(embs, list) and len(embs) == len(texts):
                        return embs
                else:
                    print(f"[embedding] batch request failed: {resp.status_code} {resp.text}")
            except Exception as e:
                print(f"[embedding] batch connection failed: {e}")

            # batch 失败：降级到单条
            print("[embedding] batch failed, fallback to single /embed requests")
            self._batch_available = False

        # 单条降级
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if text is None or len(str(text).strip()) == 0:
            return []
        try:
            resp = requests.post(
                self._single_url,
                json={"text": text},
                timeout=min(self.timeout_s, 30),
            )
            if resp.status_code == 200:
                return resp.json().get("embedding", [])
            else:
                print(f"[embedding] single request failed: {resp.status_code} {resp.text}")
                return []
        except Exception as e:
            print(f"[embedding] single request error: {e}")
            return []


# =================================================================
# 2. API 模式（原有）
# =================================================================
if (not USE_LOCAL_MODEL) and (not USE_REMOTE_SERVER):
    from langchain_openai import OpenAIEmbeddings

    class NormalizedOpenAIEmbeddings(OpenAIEmbeddings):
        def _normalize(self, embeddings: list[list[float]]) -> list[list[float]]:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            return (embeddings_array / norms).tolist()

        def embed_documents(self, texts: list[str], chunk_size: int = 0) -> list[list[float]]:
            return self._normalize(super().embed_documents(texts, chunk_size))

        def embed_query(self, text: str) -> list[float]:
            return self._normalize([super().embed_query(text)])[0]


# =================================================================
# 3. 本地模型模式（原有，很慢）
# =================================================================
if USE_LOCAL_MODEL and (not USE_REMOTE_SERVER):
    from langchain_huggingface import HuggingFaceEmbeddings

    class NormalizedLocalEmbeddings(HuggingFaceEmbeddings):
        def _normalize(self, embeddings: list[list[float]]) -> list[list[float]]:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            return (embeddings_array / norms).tolist()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._normalize(super().embed_documents(texts))

        def embed_query(self, text: str) -> list[float]:
            return self._normalize([super().embed_query(text)])[0]


def get_embedding_model():
    global _GLOBAL_EMBEDDING_MODEL
    if _GLOBAL_EMBEDDING_MODEL is not None:
        return _GLOBAL_EMBEDDING_MODEL

    # 分支 1：远程服务模式
    if USE_REMOTE_SERVER:
        print(f"[Remote Mode] connect embedding server: {BASE}")
        _GLOBAL_EMBEDDING_MODEL = RemoteEmbeddingClient(BASE, timeout_s=60)
        return _GLOBAL_EMBEDDING_MODEL

    # 分支 2：本地加载模式
    if USE_LOCAL_MODEL:
        print("[Local Mode] loading local embedding model (may be slow)...")
        _GLOBAL_EMBEDDING_MODEL = NormalizedLocalEmbeddings(
            model_name=GRAPHRAG_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},  # 或 cuda
            encode_kwargs={"normalize_embeddings": True},
        )
        return _GLOBAL_EMBEDDING_MODEL

    # 分支 3：OpenAI API 模式
    print("[API Mode] connect OpenAI-compatible embeddings...")
    _GLOBAL_EMBEDDING_MODEL = NormalizedOpenAIEmbeddings(
        model=GRAPHRAG_EMBEDDING_MODEL,
        openai_api_key=GRAPHRAG_EMBEDDING_API_KEY,
        openai_api_base=BASE,
    )
    return _GLOBAL_EMBEDDING_MODEL
