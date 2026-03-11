import os
import requests
import numpy as np
import threading
import time
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
_POOL_RAW = (os.getenv("EMBEDDING_SERVER_POOL") or "").strip()
_POOL = [p.strip().rstrip("/") for p in _POOL_RAW.split(",") if p.strip()] if _POOL_RAW else []
_SLOT_RAW = (os.getenv("EMBEDDING_SERVER_SLOT") or "").strip()
BASE = str(GRAPHRAG_EMBEDDING_API_BASE or "").strip().rstrip("/")
if _POOL:
    try:
        if _SLOT_RAW:
            idx = int(_SLOT_RAW)
        else:
            if os.getenv("USE_NEO4J_AURA", "0").strip().lower() in ("1", "true", "yes"):
                idx = 1
            else:
                idx = 0
        BASE = _POOL[idx % len(_POOL)]
    except Exception:
        BASE = _POOL[0]

# 在线embedding配置
ONLINE_API_BASE = os.getenv("GRAPHRAG_EMBEDDING_online_API_BASE", "").strip().rstrip("/")
ONLINE_API_KEY = os.getenv("GRAPHRAG_EMBEDDING_online_API_KEY", "").strip()
ONLINE_MODEL = os.getenv("GRAPHRAG_EMBEDDING_online_MODEL", "Qwen/Qwen3-Embedding-0.6B").strip()

# 是否启用混合模式（本地+在线负载均衡）
ENABLE_HYBRID_EMBEDDING = os.getenv("ENABLE_HYBRID_EMBEDDING", "1").strip().lower() in ("1", "true", "yes")

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
    带重试机制：超时或失败时自动重试
    """
    def __init__(self, base_url: str, timeout_s: int = 60, max_retries: int = 3, retry_delay: float = 1.0):
        self.base_url = (base_url or "").strip().rstrip("/")
        self.timeout_s = int(timeout_s)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

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
            for attempt in range(self.max_retries):
                try:
                    resp = requests.post(
                        self._batch_url,
                        json={"texts": texts},
                        timeout=self.timeout_s,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        embs = data.get("embeddings", [])
                        # 简单校验：长度要对得上，且每个 embedding 非空
                        if isinstance(embs, list) and len(embs) == len(texts) and all(len(e) > 0 for e in embs):
                            return embs
                        else:
                            print(f"[embedding] batch attempt {attempt+1}/{self.max_retries}: invalid response length")
                    else:
                        print(f"[embedding] batch attempt {attempt+1}/{self.max_retries}: {resp.status_code} {resp.text[:100]}")
                except requests.exceptions.Timeout:
                    print(f"[embedding] batch attempt {attempt+1}/{self.max_retries}: timeout after {self.timeout_s}s")
                except Exception as e:
                    print(f"[embedding] batch attempt {attempt+1}/{self.max_retries}: {e}")
                
                # 重试前等待
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))

            # batch 全部失败：降级到单条
            print("[embedding] batch all attempts failed, fallback to single /embed requests")
            self._batch_available = False

        # 单条降级
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if text is None or len(str(text).strip()) == 0:
            return []
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self._single_url,
                    json={"text": text},
                    timeout=min(self.timeout_s, 30),
                )
                if resp.status_code == 200:
                    emb = resp.json().get("embedding", [])
                    if emb and len(emb) > 0:
                        return emb
                    else:
                        last_error = "empty embedding returned"
                else:
                    last_error = f"{resp.status_code} {resp.text[:100]}"
            except requests.exceptions.Timeout:
                last_error = f"timeout after {min(self.timeout_s, 30)}s"
            except Exception as e:
                last_error = str(e)
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (2 ** attempt))
        
        print(f"[embedding] single request all {self.max_retries} attempts failed: {last_error}")
        return []


# =================================================================
# 1.5 在线API Embedding客户端（硅基流动等OpenAI兼容API）
# =================================================================
class OnlineAPIEmbeddingClient(Embeddings):
    """
    连接在线Embedding API（OpenAI兼容格式，如硅基流动）
    带重试机制：超时或失败时自动重试
    支持长文本截断（避免 token 限制）
    """
    def __init__(self, api_base: str, api_key: str, model: str, timeout_s: int = 30, max_retries: int = 3, retry_delay: float = 1.0, max_chars: int = 24000):
        self.api_base = (api_base or "").strip().rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = int(timeout_s)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_chars = max_chars  # BGE-M3 支持 8192 tokens，约 24000 字符
        self._url = f"{self.api_base}/embeddings"
    
    def _truncate_text(self, text: str) -> str:
        """截断文本到 max_chars 字符"""
        if len(text) <= self.max_chars:
            return text
        # 截断并保留完整的句子
        truncated = text[:self.max_chars]
        # 尝试在句号处截断
        last_period = truncated.rfind('.')
        if last_period > self.max_chars // 2:
            truncated = truncated[:last_period + 1]
        return truncated
    
    def _normalize(self, embeddings: List[List[float]]) -> List[List[float]]:
        if not embeddings:
            return embeddings
        embeddings_array = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return (embeddings_array / norms).tolist()
    
    def _request_with_retry(self, texts: List[str]) -> List[List[float]]:
        """带重试的请求逻辑"""
        # 截断长文本
        truncated_texts = [self._truncate_text(t) for t in texts]
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self._url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": truncated_texts,
                    },
                    timeout=self.timeout_s,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # OpenAI格式：{"data": [{"embedding": [...], "index": 0}, ...]}
                    embs = [item["embedding"] for item in sorted(data.get("data", []), key=lambda x: x.get("index", 0))]
                    normalized = self._normalize(embs)
                    # 验证结果有效性
                    if normalized and len(normalized) == len(texts) and all(len(e) > 0 for e in normalized):
                        return normalized
                    else:
                        last_error = f"Invalid response: got {len(normalized)} embeddings for {len(texts)} texts"
                        print(f"[online_embedding] attempt {attempt+1}/{self.max_retries}: {last_error}")
                elif resp.status_code == 429:
                    # Rate limit - 等待更长时间
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"[online_embedding] rate limited, waiting {wait_time}s before retry {attempt+1}/{self.max_retries}")
                    time.sleep(wait_time)
                    last_error = f"Rate limited: {resp.status_code}"
                elif resp.status_code >= 500:
                    # 服务器错误 - 重试
                    last_error = f"Server error: {resp.status_code} {resp.text[:100]}"
                    print(f"[online_embedding] attempt {attempt+1}/{self.max_retries}: {last_error}")
                else:
                    # 客户端错误 - 不重试
                    print(f"[online_embedding] client error (no retry): {resp.status_code} {resp.text[:200]}")
                    return []
            except requests.exceptions.Timeout as e:
                last_error = f"Timeout after {self.timeout_s}s"
                print(f"[online_embedding] attempt {attempt+1}/{self.max_retries}: {last_error}")
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {e}"
                print(f"[online_embedding] attempt {attempt+1}/{self.max_retries}: {last_error}")
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                print(f"[online_embedding] attempt {attempt+1}/{self.max_retries}: {last_error}")
            
            # 重试前等待（指数退避）
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                time.sleep(wait_time)
        
        print(f"[online_embedding] all {self.max_retries} attempts failed. Last error: {last_error}")
        return []
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._request_with_retry(texts)
    
    def embed_query(self, text: str) -> List[float]:
        if text is None or len(str(text).strip()) == 0:
            return []
        result = self.embed_documents([text])
        return result[0] if result else []


# =================================================================
# 1.6 混合负载均衡Embedding（本地+在线并行 + 本地并发保护）
# =================================================================

# 全局本地并发锁：确保本地embedding同时只处理有限请求
_LOCAL_SEMAPHORE = threading.Semaphore(int(os.getenv("LOCAL_EMBED_MAX_CONCURRENT", "1")))

# 跨进程文件锁（防止多个Worker同时调用本地Embedding）
_LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".local_embed.lock")


def _acquire_file_lock(timeout_s: float = 30.0) -> bool:
    """尝试获取文件锁，防止多进程同时调用本地Embedding"""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.1)
    return False


def _release_file_lock():
    """释放文件锁"""
    try:
        os.remove(_LOCK_FILE)
    except:
        pass


class HybridEmbeddingClient(Embeddings):
    """
    混合Embedding客户端：在线为主力，本地为补充
    
    策略：
    1. 在线处理大部分请求（85%）
    2. 本地只处理少量请求（15%），且有跨进程文件锁保护
    3. 本地每批只处理1条，避免显存爆炸
    4. 某个源失败时自动切换到另一个
    5. 当 local_ratio=0 时，完全跳过本地调用
    """
    def __init__(self, local_client: Embeddings, online_client: Embeddings, local_ratio: float = 0.15):
        self.local_client = local_client
        self.online_client = online_client
        self.local_ratio = min(0.3, local_ratio)  # 本地最多30%
        self._counter = 0
        self._lock = threading.Lock()
        self._online_fail_count = 0
        self._online_fail_threshold = 3
        self._online_disabled_until = 0
        self._local_batch_size = 1  # 本地每批只处理1条，最稳定
        self._local_disabled = (local_ratio <= 0)  # 当 local_ratio=0 时禁用本地
    
    def _should_use_local(self) -> bool:
        """决定是否使用本地（概率性）"""
        if self._local_disabled:
            return False
        if time.time() < self._online_disabled_until:
            return True
        with self._lock:
            self._counter += 1
            return (self._counter % 100) < (self.local_ratio * 100)
    
    def _mark_online_success(self):
        self._online_fail_count = 0
    
    def _mark_online_fail(self):
        self._online_fail_count += 1
        if self._online_fail_count >= self._online_fail_threshold:
            self._online_disabled_until = time.time() + 60
            print(f"[hybrid_embedding] 在线API连续失败{self._online_fail_count}次，暂时禁用60秒")
    
    def _local_embed_with_protection(self, text: str) -> List[float]:
        """带跨进程保护的本地单条embedding"""
        if not _acquire_file_lock(timeout_s=10):
            # 获取锁失败，直接用在线
            return self.online_client.embed_query(text)
        try:
            with _LOCAL_SEMAPHORE:
                return self.local_client.embed_query(text)
        finally:
            _release_file_lock()
    
    def _local_embed_batch_with_protection(self, texts: List[str]) -> List[List[float]]:
        """带跨进程保护的本地批量embedding，逐条处理"""
        results = []
        for t in texts:
            try:
                result = self._local_embed_with_protection(t)
                results.append(result if result else [])
            except Exception as e:
                print(f"[hybrid_embedding] 本地单条失败: {e}")
                # 失败时用在线补救
                try:
                    results.append(self.online_client.embed_query(t))
                except:
                    results.append([])
        return results
    
    def embed_query(self, text: str) -> List[float]:
        if text is None or len(str(text).strip()) == 0:
            return []
        
        use_local = self._should_use_local()
        
        if use_local and not self._local_disabled:
            try:
                result = self._local_embed_with_protection(text)
                if result:
                    return result
            except Exception as e:
                pass  # 静默失败，不打印错误
            # 本地失败，尝试在线
        
        # 在线处理
        try:
            result = self.online_client.embed_query(text)
            if result:
                self._mark_online_success()
                return result
        except Exception as e:
            print(f"[hybrid_embedding] 在线失败: {e}")
            self._mark_online_fail()
        
        # 都失败了，最后尝试本地（仅当本地未禁用时）
        if not use_local and not self._local_disabled:
            try:
                return self._local_embed_with_protection(text)
            except:
                pass
        return []
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        n = len(texts)
        
        # 当本地禁用时，全部用在线
        if self._local_disabled:
            try:
                result = self.online_client.embed_documents(texts)
                if result and len(result) == n and all(len(e) > 0 for e in result):
                    self._mark_online_success()
                    return result
                else:
                    print(f"[hybrid_embedding] 在线返回无效: result_len={len(result) if result else 0}, expected={n}, valid={sum(1 for e in (result or []) if len(e) > 0)}")
            except Exception as e:
                print(f"[hybrid_embedding] 在线批量失败: {e}")
                self._mark_online_fail()
            return [[] for _ in texts]
        
        # 小批量：直接用在线
        if n <= 5:
            try:
                result = self.online_client.embed_documents(texts)
                if result and len(result) == n:
                    self._mark_online_success()
                    return result
            except Exception as e:
                print(f"[hybrid_embedding] 在线批量失败: {e}")
                self._mark_online_fail()
            # 在线失败，用本地逐条处理
            return self._local_embed_batch_with_protection(texts)
        
        # 大批量：在线为主，本地为辅
        local_count = max(1, int(n * self.local_ratio)) if not self._local_disabled else 0
        local_texts = texts[:local_count]
        online_texts = texts[local_count:]
        
        local_results = []
        online_results = []
        online_error = None
        
        def do_local():
            nonlocal local_results
            if not local_texts or self._local_disabled:
                return
            local_results = self._local_embed_batch_with_protection(local_texts)
        
        def do_online():
            nonlocal online_results, online_error
            if not online_texts:
                return
            if time.time() < self._online_disabled_until and not self._local_disabled:
                # 在线被禁用，用本地处理
                online_results = self._local_embed_batch_with_protection(online_texts)
                return
            try:
                online_results = self.online_client.embed_documents(online_texts)
                if online_results:
                    self._mark_online_success()
            except Exception as e:
                online_error = e
                print(f"[hybrid_embedding] 在线并行失败: {e}")
                self._mark_online_fail()
        
        # 并行执行
        t1 = threading.Thread(target=do_local)
        t2 = threading.Thread(target=do_online)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        # 在线失败时用本地补救（仅当本地未禁用时）
        if online_error and not online_results and not self._local_disabled:
            online_results = self._local_embed_batch_with_protection(online_texts)
        
        # 确保结果长度正确
        if len(local_results) != len(local_texts):
            local_results = [[] for _ in local_texts]
        if len(online_results) != len(online_texts):
            online_results = [[] for _ in online_texts]
        
        return local_results + online_results


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

    # 分支 0：混合模式（本地+在线负载均衡）
    if ENABLE_HYBRID_EMBEDDING and USE_REMOTE_SERVER and ONLINE_API_BASE and ONLINE_API_KEY:
        print(f"[Hybrid Mode] 本地: {BASE} + 在线: {ONLINE_API_BASE}")
        local_client = RemoteEmbeddingClient(BASE, timeout_s=60)
        online_client = OnlineAPIEmbeddingClient(
            api_base=ONLINE_API_BASE,
            api_key=ONLINE_API_KEY,
            model=ONLINE_MODEL,
            timeout_s=30,
        )
        # 本地占50%，在线占50%
        local_ratio = float(os.getenv("HYBRID_LOCAL_RATIO", "0.5"))
        _GLOBAL_EMBEDDING_MODEL = HybridEmbeddingClient(local_client, online_client, local_ratio=local_ratio)
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
