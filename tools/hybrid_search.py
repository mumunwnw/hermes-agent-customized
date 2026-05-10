#!/usr/bin/env python3
"""Hybrid Search - BM25 + Vector + RRF for session search.

This module provides hybrid search capability by combining:
1. BM25 keyword search (FTS5 - already in state.db)
2. Vector semantic search (sqlite-vec extension)
3. RRF (Reciprocal Rank Fusion) for result merging
4. Optional reranker for final ranking

Usage:
    from tools.hybrid_search import HybridSessionSearch
    
    search = HybridSessionSearch(db)
    results = search.search("部署数据库的问题", limit=10)
"""

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_hybrid_instance = None
_hybrid_lock = threading.Lock()
_thread_local = threading.local()


def _parse_interval(value) -> int:
    """Parse time interval config to seconds.
    
    Accepts:
      - int/float: raw seconds (30 → 30)
      - str with units: "30s", "15min", "2hr", "1h30m"
      - str plain number: "30" → 30
    Returns seconds as int. Minimum 5.
    """
    if isinstance(value, (int, float)):
        return max(5, int(value))
    
    if not isinstance(value, str):
        return 30
    
    s = value.strip().lower()
    if not s:
        return 30
    
    try:
        return max(5, int(s))
    except ValueError:
        pass
    
    import re
    total = 0
    found = False
    for match in re.finditer(r'(\d+(?:\.\d+)?)\s*(hr|h|hour|min|m|minute|s|sec|second)', s):
        amount = float(match.group(1))
        unit = match.group(2)
        if unit in ('hr', 'h', 'hour'):
            total += amount * 3600
        elif unit in ('min', 'm', 'minute'):
            total += amount * 60
        elif unit in ('s', 'sec', 'second'):
            total += amount
        found = True
    
    if not found:
        try:
            total = int(s)
        except ValueError:
            return 30
    
    return max(5, int(total))


def _vec_serialize(vector: List[float]) -> Optional[bytes]:
    """Serialize float vector for sqlite-vec. Handles API name differences.
    Returns None if serialization fails (e.g. vector contains None)."""
    if vector is None:
        return None
    try:
        import sqlite_vec
        fn = getattr(sqlite_vec, 'serialize_float32', None) or sqlite_vec.serialize_f32
        return fn(vector)
    except (TypeError, AttributeError, ValueError) as e:
        logger.warning("Failed to serialize vector: %s", e)
        return None


def _check_sqlite_vec_available() -> bool:
    """Check if sqlite-vec extension is available."""
    try:
        import sqlite_vec
        return True
    except ImportError:
        return False


def _detect_embedding_dim(api_url: str, model: str, api_key: str) -> int:
    """Probe embedding API to detect output dimension by sending a test string."""
    if not api_key:
        return 1024
    try:
        import httpx
        response = httpx.post(
            f"{api_url}/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input": "test"},
            timeout=15,
        )
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data and "embedding" in data[0]:
                return len(data[0]["embedding"])
    except Exception as e:
        logger.debug("Could not detect embedding dimension: %s", e)
    return 1024


class EmbeddingIndexer:
    """Embedding 索引器 - 批量模式，由守护线程驱动。"""
    
    _MODEL_MAX_TOKENS = 8192
    
    def __init__(self, db, api_url: str, model: str, api_key: str,
                 min_content_length: int = None, batch_token_limit: int = 7000,
                 embedding_dim: int = 1024):
        self.db = db
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.min_content_length = min_content_length
        self.batch_token_limit = batch_token_limit
        self.embedding_dim = embedding_dim
        
        self._count_lock = threading.Lock()
        self._indexed_count = 0
        self._failed_count = 0
    
    @property
    def indexed_count(self):
        with self._count_lock:
            return self._indexed_count
    
    @property
    def failed_count(self):
        with self._count_lock:
            return self._failed_count
    
    def _increment_indexed(self, n=1):
        with self._count_lock:
            self._indexed_count += n
    
    def _increment_failed(self, n=1):
        with self._count_lock:
            self._failed_count += n
    
    @staticmethod
    def _estimate_tokens(content: str, token_count: int = None) -> int:
        if token_count is not None and token_count > 0:
            return token_count
        if not content:
            return 0
        cjk = sum(1 for ch in content if '\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff')
        return int(len(content) + cjk * 0.5)
    
    def _truncate_to_token_budget(self, content: str, token_budget: int) -> str:
        estimated = self._estimate_tokens(content)
        if estimated <= token_budget:
            return content
        ratio = token_budget / estimated
        char_limit = max(1, int(len(content) * ratio * 0.9))
        truncated = content[:char_limit]
        try:
            truncated = truncated.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            truncated = content[:max(1, char_limit - 100)]
        return truncated
    
    def index_batch(self, items: list, stop_event: threading.Event = None):
        """批量索引：按 token 分组，每组一次 API 调用 + 一次 DB 写入。
        
        items: [(message_id, content, session_id, token_count), ...]
        stop_event: if set, abort between batches
        """
        if not items or not self.api_key:
            return
        
        batch_start_indexed = self.indexed_count
        batch_start_failed = self.failed_count
        
        import sqlite_vec
        
        db_path = getattr(self.db, 'db_path', None) or getattr(self.db, '_db_path', None)
        if not db_path:
            try:
                row = self.db._conn.execute("PRAGMA database_list").fetchone()
                if row:
                    db_path = row["file"] if hasattr(row, "keys") else row[2]
            except Exception:
                pass
        if not db_path or db_path == "":
            logger.warning("Cannot determine db path for batch indexing")
            return
        
        conn = None
        try:
            from pysqlite3 import dbapi2 as pysqlite
            conn = pysqlite.connect(db_path, check_same_thread=False)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except ImportError:
            import sqlite3
            conn = sqlite3.connect(db_path)
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except Exception:
                vec_path = sqlite_vec.loadable_path()
                if callable(vec_path):
                    vec_path = vec_path()
                import sys
                if sys.platform == "darwin" and not str(vec_path).endswith(".dylib"):
                    vec_path = str(vec_path) + ".dylib"
                conn.load_extension(str(vec_path))
                try:
                    conn.enable_load_extension(False)
                except AttributeError:
                    pass
        
        try:
            vec_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
            ).fetchone()
            if not vec_exists:
                conn.execute(
                    f"""CREATE VIRTUAL TABLE IF NOT EXISTS message_vec USING vec0(
                        message_id INTEGER PRIMARY KEY,
                        embedding FLOAT[{self.embedding_dim}],
                        session_id TEXT
                    )"""
                )
                conn.commit()
            
            batches = self._group_into_batches(items)
            
            for batch in batches:
                if stop_event and stop_event.is_set():
                    logger.info("Index batch: aborted → stop event received")
                    break
                
                texts = []
                valid_items = []
                for msg_id, content, sid, tc in batch:
                    text = self._truncate_to_token_budget(content, self._MODEL_MAX_TOKENS)
                    if text and text.strip():
                        texts.append(text)
                        valid_items.append((msg_id, sid))
                
                if not texts:
                    continue
                
                embeddings = self._call_embedding_api_batch(texts)
                if not embeddings or len(embeddings) != len(valid_items):
                    self._increment_failed(len(valid_items))
                    continue
                
                for (msg_id, sid), emb in zip(valid_items, embeddings):
                    if emb is None:
                        self._increment_failed()
                        continue
                    serialized = _vec_serialize(emb)
                    if serialized is None:
                        self._increment_failed()
                        continue
                    try:
                        conn.execute(
                            """INSERT OR REPLACE INTO message_vec 
                               (message_id, embedding, session_id) 
                               VALUES (?, ?, ?)""",
                            (msg_id, serialized, sid)
                        )
                        self._increment_indexed()
                    except Exception as e:
                        logger.warning("Failed to insert embedding for %d: %s", msg_id, e)
                        self._increment_failed()
                
                conn.commit()
            
            logger.debug(
                "Index batch (embedding): %d items → %d indexed, %d failed, %d skipped",
                len(items),
                self.indexed_count - batch_start_indexed,
                self.failed_count - batch_start_failed,
                len(items) - (self.indexed_count - batch_start_indexed) - (self.failed_count - batch_start_failed),
            )
        finally:
            if conn:
                conn.close()
    
    def _group_into_batches(self, items: list) -> list:
        """按 token 预算分组。每组总 token 数不超过 batch_token_limit。"""
        batches = []
        current_batch = []
        current_tokens = 0
        
        for item in items:
            msg_id, content, sid, tc = item
            est = self._estimate_tokens(content, tc)
            
            if est > self._MODEL_MAX_TOKENS:
                est = int(self._MODEL_MAX_TOKENS * 0.9)
            
            if current_batch and current_tokens + est > self.batch_token_limit:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(item)
            current_tokens += est
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _call_embedding_api_batch(self, texts: list, max_retries: int = 3) -> Optional[list]:
        """批量调用 embedding API。Returns list of embeddings or None."""
        if not self.api_key or not texts:
            return None
        
        import httpx
        
        for attempt in range(max_retries):
            try:
                response = httpx.post(
                    f"{self.api_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "input": texts,
                    },
                    timeout=60
                )
                
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning("Embedding API rate limited (batch), waiting %ds", wait_time)
                    time.sleep(wait_time)
                    continue
                
                if response.status_code == 400:
                    logger.warning("Embedding API 400 error for batch size=%d, falling back to single",
                                   len(texts))
                    return self._fallback_single_calls(texts)
                
                response.raise_for_status()
                data = response.json().get("data", [])
                if not data or len(data) != len(texts):
                    logger.warning("Embedding API returned %d results for %d inputs",
                                   len(data), len(texts))
                    return self._fallback_single_calls(texts)
                try:
                    data.sort(key=lambda x: x.get("index", 0))
                except (KeyError, TypeError):
                    pass
                embeddings = []
                for d in data:
                    emb = d.get("embedding")
                    if emb is None:
                        embeddings.append(None)
                    else:
                        embeddings.append(emb)
                return embeddings
            
            except httpx.TimeoutException:
                logger.warning("Embedding API timeout (batch attempt %d/%d)", attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            
            except Exception as e:
                logger.error("Embedding API batch error: %s", e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def _fallback_single_calls(self, texts: list) -> list:
        """批量失败时回退到逐条调用。None for failed items."""
        results = []
        for text in texts:
            emb = self._call_embedding_api_with_retry(text)
            results.append(emb)
        return results
    
    def _call_embedding_api_with_retry(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """带重试的单条 embedding API 调用。"""
        if not self.api_key:
            return None
        
        if not text or not text.strip():
            return None
        
        import httpx
        
        for attempt in range(max_retries):
            try:
                response = httpx.post(
                    f"{self.api_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "input": text,
                    },
                    timeout=30
                )
                
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning("Embedding API rate limited, waiting %ds", wait_time)
                    time.sleep(wait_time)
                    continue
                
                if response.status_code == 400:
                    logger.warning("Embedding API 400 error for text length=%d (first 100 chars): %s",
                                   len(text), text[:100])
                    return None
                
                response.raise_for_status()
                data = response.json().get("data", [])
                if data and "embedding" in data[0]:
                    return data[0]["embedding"]
                return None
            
            except httpx.TimeoutException:
                logger.warning("Embedding API timeout (attempt %d/%d)", attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            
            except Exception as e:
                logger.error("Embedding API error: %s", e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None

class HybridSessionSearch:
    """混合搜索：BM25 + Vector + RRF。"""
    
    def __init__(self, db, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self._db_lock = getattr(db, '_lock', None) or threading.Lock()
        
        hybrid_config = self.config.get("hybrid", {})
        
        # Embedding 配置（独立于 LLM）
        self.embedding_api_url = (
            hybrid_config.get("embed_base_url") or
            "https://api.siliconflow.cn/v1"
        )
        self.embedding_model = hybrid_config.get("embed_model", "BAAI/bge-m3")
        self.api_key = (
            hybrid_config.get("embed_api_key") or
            os.environ.get("SILICONFLOW_API_KEY", "")
        )
        self.timeout = self.config.get("timeout", 30)
        
        # Reranker 配置
        self.use_reranker = hybrid_config.get("use_reranker", False)
        self.reranker_model = hybrid_config.get("reranker_model", "BAAI/bge-reranker-v2-m3")
        self.reranker_api_key = (
            hybrid_config.get("reranker_api_key") or
            self.api_key
        )
        self.reranker_api_url = (
            hybrid_config.get("reranker_base_url") or
            self.embedding_api_url
        )
        self.rrf_k = hybrid_config.get("rrf_k", 60)
        self.vec_top_k = hybrid_config.get("vec_top_k", 50)
        self.vec_distance_threshold = hybrid_config.get("vec_distance_threshold", 1.2)
        self.rrf_score_threshold = hybrid_config.get("rrf_score_threshold", 0.0)
        self.index_roles = hybrid_config.get("index_roles", ["user", "assistant"])
        _raw_min = hybrid_config.get("min_content_length", 0)
        self.min_content_length = int(_raw_min) if _raw_min is not None else 0
        self.batch_token_limit = hybrid_config.get("batch_token_limit", 7000)
        self.auto_index = hybrid_config.get("auto_index", True)
        self._indexing_lock = threading.Lock()
        self._idle_index_interval = _parse_interval(hybrid_config.get("idle_index_interval", "15min"))
        self._stop_event = threading.Event()
        self._last_index_time = 0.0
        
        # Detect embedding dimension from config or probe API
        self.embedding_dim = hybrid_config.get("embed_dim", 0)
        if not self.embedding_dim or self.embedding_dim <= 0:
            self.embedding_dim = _detect_embedding_dim(
                self.embedding_api_url, self.embedding_model, self.api_key
            )
        
        # 检查 sqlite-vec 可用性并加载扩展
        self.vec_available = self._load_vec_extension()
        if not self.vec_available:
            logger.warning("sqlite-vec not available, vector search disabled")
        
        # 初始化索引器
        self.indexer = EmbeddingIndexer(
            db, self.embedding_api_url, self.embedding_model, self.api_key,
            min_content_length=self.min_content_length,
            batch_token_limit=self.batch_token_limit,
            embedding_dim=self.embedding_dim,
        )
        


    def _role_filter_sql(self, table_alias: str = "m") -> tuple:
        return (
            f"AND {table_alias}.role IN ({','.join('?' for _ in self.index_roles)}) ",
            list(self.index_roles),
        )

    def _min_length_sql(self, table_alias: str = "m") -> tuple:
        if self.min_content_length and self.min_content_length > 0:
            return (f"AND LENGTH({table_alias}.content) >= ? ", [self.min_content_length])
        return ("", [])
    
    def _load_vec_extension(self) -> bool:
        """尝试加载 sqlite-vec 扩展到数据库连接。"""
        try:
            import sqlite_vec
            
            conn = self.db._conn
            
            # 方法 1: sqlite_vec.load() — 推荐，macOS 兼容
            try:
                conn.enable_load_extension(True)
            except AttributeError:
                pass
            
            try:
                sqlite_vec.load(conn)
                try:
                    conn.enable_load_extension(False)
                except AttributeError:
                    pass
                logger.debug("sqlite-vec extension loaded via sqlite_vec.load()")
                return True
            except Exception:
                pass
            
            # 方法 2: load_extension + loadable_path — Linux/Windows
            try:
                vec_path = sqlite_vec.loadable_path()
                if callable(vec_path):
                    vec_path = vec_path()
                
                import sys
                if sys.platform == "darwin" and not str(vec_path).endswith(".dylib"):
                    vec_path = str(vec_path) + ".dylib"
                
                conn.load_extension(str(vec_path))
                
                try:
                    conn.enable_load_extension(False)
                except AttributeError:
                    pass
                
                logger.debug("sqlite-vec extension loaded via load_extension")
                return True
            except Exception:
                pass
            
            # 方法 3: 替换连接为 pysqlite3 — macOS 最后手段
            # WARNING: This replaces self.db._conn which other threads may be using.
            # We hold _db_lock to minimize the race window, but callers should
            # ensure no other DB operations are in flight when HybridSessionSearch
            # is first constructed.
            try:
                from pysqlite3 import dbapi2 as pysqlite
                db_path = None
                try:
                    row = conn.execute("PRAGMA database_list").fetchone()
                    if row:
                        db_path = row["file"] if hasattr(row, "keys") else row[2]
                except Exception:
                    pass
                
                if not db_path:
                    logger.warning("Cannot determine db_path for pysqlite3 connection swap, skipping")
                    return False
                
                new_conn = pysqlite.connect(db_path, check_same_thread=False)
                new_conn.enable_load_extension(True)
                sqlite_vec.load(new_conn)
                new_conn.enable_load_extension(False)
                new_conn.row_factory = pysqlite.Row
                new_conn.execute("PRAGMA journal_mode=WAL")
                new_conn.execute("PRAGMA foreign_keys=ON")
                
                with self._db_lock:
                    old_conn = self.db._conn
                    self.db._conn = new_conn
                try:
                    old_conn.close()
                except Exception:
                    pass
                logger.debug("sqlite-vec loaded via pysqlite3 connection swap")
                return True
            except Exception:
                pass
            
            logger.debug("All sqlite-vec loading methods failed")
            return False
        except ImportError:
            logger.debug("sqlite-vec not installed")
            return False
    
    def _call_embedding_api(self, text: str) -> Optional[List[float]]:
        """调用 embedding API（带重试）。"""
        return self.indexer._call_embedding_api_with_retry(text)
    
    def _bm25_search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """BM25 关键词搜索（使用 FTS5）。"""
        conn = self._thread_safe_conn()
        if conn is None:
            return []
        try:
            role_clause, role_params = self._role_filter_sql()
            min_len_clause, min_len_params = self._min_length_sql()
            
            fts_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()
            if not fts_exists:
                return []
            
            results = conn.execute(
                f"""SELECT m.id, m.session_id, m.content, m.role, m.timestamp,
                           bm25(messages_fts) as rank
                    FROM messages_fts f
                    JOIN messages m ON m.id = f.rowid
                    JOIN sessions s ON m.session_id = s.id
                    WHERE messages_fts MATCH ?
                    {role_clause}
                    {min_len_clause}
                    AND (s.source IS NULL OR s.source NOT IN ('tool'))
                    ORDER BY rank
                    LIMIT ?""",
                [query] + role_params + min_len_params + [limit]
            ).fetchall()
            
            formatted = []
            for r in results:
                content = r[2] or ""
                formatted.append({
                    "message_id": r[0],
                    "session_id": r[1],
                    "content": content,
                    "role": r[3],
                    "timestamp": r[4],
                    "snippet": content[:200] + "...",
                    "bm25_rank": r[5],
                })
            
            return formatted
        except Exception as e:
            logger.warning("BM25 search failed: %s", e, exc_info=True)
            return []
    
    def _ensure_vec_loaded(self):
        """DEPRECATED: Use _thread_safe_conn() instead.
        
        This method operates on self.db._conn which is not thread-safe.
        Kept only for backward compatibility — do not call from new code.
        """
        return False

    def _vector_search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """向量语义搜索（使用 sqlite-vec）。"""
        if not self.vec_available:
            return []
        
        conn = self._thread_safe_conn()
        if conn is None:
            return []
        
        query_embedding = self._call_embedding_api(query)
        if not query_embedding:
            return []
        
        query_vec = _vec_serialize(query_embedding)
        if query_vec is None:
            logger.warning("Failed to serialize query embedding for vector search")
            return []
        
        try:
            vec_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
            ).fetchone()
            
            if not vec_exists:
                return []
            
            top_k = min(self.vec_top_k, limit)
            
            role_clause, role_params = self._role_filter_sql()
            
            results = conn.execute(
                f"""SELECT m.id, m.session_id, m.content, m.role, m.timestamp,
                          distance
                   FROM message_vec
                   JOIN messages m ON m.id = message_vec.message_id
                   WHERE embedding MATCH ?
                     AND k = ?
                     {role_clause}
                   ORDER BY distance""",
                [query_vec, top_k] + role_params
            ).fetchall()
            
            formatted = []
            for r in results:
                distance = r[5]
                if distance is None:
                    continue
                content = r[2] or ""
                formatted.append({
                    "message_id": r[0],
                    "session_id": r[1],
                    "content": content,
                    "role": r[3],
                    "timestamp": r[4],
                    "snippet": content[:200] + "...",
                    "vector_distance": distance,
                })
            
            return formatted
        except Exception as e:
            logger.warning("Vector search failed: %s", e, exc_info=True)
            return []
    
    def _rrf_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """RRF 融合 - 合并 BM25 和 Vector 结果。
        
        When a message appears in both result sets, we merge metadata
        from both sources rather than letting one overwrite the other.
        """
        from collections import defaultdict
        
        scores = defaultdict(float)
        result_map = {}
        
        # BM25 排名贡献
        for rank, result in enumerate(bm25_results, 1):
            msg_id = result["message_id"]
            scores[msg_id] += 1.0 / (self.rrf_k + rank)
            if msg_id not in result_map:
                result_map[msg_id] = result.copy()
            else:
                result_map[msg_id].update({
                    k: v for k, v in result.items()
                    if k not in ("vector_distance", "rrf_score")
                })
        
        # Vector 排名贡献 — merge rather than overwrite
        for rank, result in enumerate(vector_results, 1):
            msg_id = result["message_id"]
            scores[msg_id] += 1.0 / (self.rrf_k + rank)
            if msg_id not in result_map:
                result_map[msg_id] = result.copy()
            else:
                # Preserve BM25 snippet (usually higher quality) and add vector_distance
                result_map[msg_id]["vector_distance"] = result.get("vector_distance")
                if "snippet" not in result_map[msg_id] or not result_map[msg_id]["snippet"]:
                    result_map[msg_id]["snippet"] = result.get("snippet", "")
        
        # 按融合分数排序
        fused = []
        for msg_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]:
            result = result_map[msg_id]
            result["rrf_score"] = score
            fused.append(result)
        
        return fused
    
    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        
        query = query.strip()
        search_start = time.monotonic()
        
        bm25_results = self._bm25_search(query, limit=50)
        vector_results_raw = self._vector_search(query, limit=50)
        vec_before_threshold = len(vector_results_raw)
        
        vector_results = [r for r in vector_results_raw
                          if r.get("vector_distance") is not None and r["vector_distance"] <= self.vec_distance_threshold]
        
        fused_results = self._rrf_fusion(bm25_results, vector_results, limit=limit * 2)
        fused_before_threshold = len(fused_results)
        
        if self.rrf_score_threshold > 0 and fused_results:
            fused_results = [r for r in fused_results if r.get("rrf_score", 0) >= self.rrf_score_threshold]
        
        if self.use_reranker and fused_results:
            fused_results = self._rerank_results(query, fused_results, limit=limit)
        
        elapsed = time.monotonic() - search_start
        
        pipeline = f"bm25={len(bm25_results)}"
        if vec_before_threshold != len(vector_results):
            pipeline += f" → vec_raw={vec_before_threshold} → vec_filtered={len(vector_results)}(thresh={self.vec_distance_threshold})"
        else:
            pipeline += f" → vec={len(vector_results)}"
        pipeline += f" → fused={fused_before_threshold}"
        if self.rrf_score_threshold > 0:
            pipeline += f" → rrf_filtered={len(fused_results)}(thresh={self.rrf_score_threshold})"
        if self.use_reranker:
            pipeline += f" → reranked={len(fused_results)}"
        pipeline += f" → result={min(len(fused_results), limit)}"
        
        logger.info(
            "Hybrid search | query=%r | %s | %.2fs",
            query[:80], pipeline, elapsed,
        )
        
        for r in fused_results[:limit]:
            r["_diagnostics"] = {
                "bm25_hits": len(bm25_results),
                "vector_hits": len(vector_results),
                "vec_before_threshold": vec_before_threshold,
                "vec_distance_threshold": self.vec_distance_threshold,
                "fused_before_rrf_threshold": fused_before_threshold,
                "fused_after_rrf_threshold": len(fused_results),
                "rrf_score_threshold": self.rrf_score_threshold,
                "vec_available": self.vec_available,
                "has_api_key": bool(self.api_key),
            }
        
        return fused_results[:limit]
    
    def _rerank_results(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """使用远程 reranker API 重排序。"""
        if not candidates or not self.reranker_api_key:
            return candidates
        
        try:
            import httpx
            
            # 准备文档
            documents = [
                f"[{c['role']}] {c.get('snippet', c.get('content', ''))[:500]}"
                for c in candidates[:20]
            ]
            
            # 调用 reranker API
            response = httpx.post(
                f"{self.reranker_api_url}/rerank",
                headers={"Authorization": f"Bearer {self.reranker_api_key}"},
                json={
                    "model": self.reranker_model,
                    "query": query,
                    "documents": documents,
                },
                timeout=30
            )
            response.raise_for_status()
            
            # 按 rerank 分数重新排序
            reranked = []
            for hit in response.json()["results"]:
                idx = hit["index"]
                if idx < 0 or idx >= len(candidates):
                    logger.warning("Reranker returned out-of-range index %d (candidates=%d)", idx, len(candidates))
                    continue
                result = candidates[idx].copy()
                result["rerank_score"] = hit["relevance_score"]
                reranked.append(result)
            
            return reranked[:limit]
        except Exception as e:
            logger.warning("Reranker failed: %s", e, exc_info=True)
            return candidates[:limit]
    
    def _thread_safe_conn(self):
        """Create a thread-safe read connection with sqlite-vec loaded.
        
        Uses a per-thread cached connection to avoid repeated extension
        loading overhead during frequent searches. The cached connection
        is stored in _thread_local and reused within the same thread.
        Caller must NOT close the returned connection — it is managed
        by this method.
        Returns None if db path cannot be determined.
        """
        cached = getattr(_thread_local, 'hybrid_conn', None)
        if cached is not None:
            try:
                cached.execute("SELECT 1").fetchone()
                if self.vec_available:
                    cached.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
                    ).fetchone()
                return cached
            except Exception:
                try:
                    cached.close()
                except Exception:
                    pass
                _thread_local.hybrid_conn = None
        
        db_path = getattr(self.db, 'db_path', None) or getattr(self.db, '_db_path', None)
        if not db_path:
            try:
                from hermes_constants import get_hermes_home
                candidate = get_hermes_home() / "state.db"
                if candidate.exists():
                    db_path = str(candidate)
            except Exception:
                pass
        if not db_path:
            try:
                row = self.db._conn.execute("PRAGMA database_list").fetchone()
                if row:
                    db_path = row["file"] if hasattr(row, "keys") else row[2]
            except Exception:
                pass
        if not db_path:
            return None
        
        conn = None
        vec_loaded = False
        
        if self.vec_available:
            try:
                import sqlite_vec
                
                # Method 1: pysqlite3 (most reliable on macOS)
                try:
                    from pysqlite3 import dbapi2 as pysqlite
                    conn = pysqlite.connect(str(db_path), check_same_thread=False)
                    conn.enable_load_extension(True)
                    sqlite_vec.load(conn)
                    conn.enable_load_extension(False)
                    conn.row_factory = pysqlite.Row
                    vec_loaded = True
                except Exception:
                    if conn:
                        try:
                            conn.close()
                        except Exception:
                            pass
                    conn = None
                
                # Method 2: standard sqlite3 + sqlite_vec.load()
                if not vec_loaded:
                    try:
                        import sqlite3
                        conn = sqlite3.connect(str(db_path))
                        conn.row_factory = sqlite3.Row
                        conn.enable_load_extension(True)
                        sqlite_vec.load(conn)
                        conn.enable_load_extension(False)
                        vec_loaded = True
                    except Exception:
                        if conn:
                            try:
                                conn.close()
                            except Exception:
                                pass
                        conn = None
                
                # Method 3: standard sqlite3 + loadable_path
                if not vec_loaded:
                    try:
                        import sqlite3
                        conn = sqlite3.connect(str(db_path))
                        conn.row_factory = sqlite3.Row
                        vec_path = sqlite_vec.loadable_path()
                        if callable(vec_path):
                            vec_path = vec_path()
                        import sys
                        if sys.platform == "darwin" and not str(vec_path).endswith(".dylib"):
                            vec_path = str(vec_path) + ".dylib"
                        conn.enable_load_extension(True)
                        conn.load_extension(str(vec_path))
                        conn.enable_load_extension(False)
                        vec_loaded = True
                    except Exception:
                        if conn:
                            try:
                                conn.close()
                            except Exception:
                                pass
                        conn = None
            except ImportError:
                pass
        
        if conn is None:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
        
        _thread_local.hybrid_conn = conn
        return conn

    def _fetch_unindexed(self, limit: int = None) -> list:
        if not self.vec_available or not self.api_key:
            return []
        
        conn = self._thread_safe_conn()
        if conn is None:
            return []
        
        try:
            role_clause, role_params = self._role_filter_sql()
            min_len_clause, min_len_params = self._min_length_sql()
            limit_clause = f"LIMIT ? " if limit else ""
            limit_params = [limit] if limit else []
            
            vec_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
            ).fetchone()
            
            if not vec_exists:
                rows = conn.execute(
                    f"""SELECT m.id, m.content, m.session_id, m.token_count FROM messages m
                       WHERE m.content IS NOT NULL AND m.content <> '' {min_len_clause}
                       {role_clause}
                       ORDER BY m.timestamp DESC {limit_clause}""",
                    min_len_params + role_params + limit_params
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""SELECT m.id, m.content, m.session_id, m.token_count FROM messages m
                       LEFT JOIN message_vec v ON m.id = v.message_id
                       WHERE v.message_id IS NULL
                         AND m.content IS NOT NULL AND m.content <> '' {min_len_clause}
                       {role_clause}
                       ORDER BY m.timestamp DESC {limit_clause}""",
                    min_len_params + role_params + limit_params
                ).fetchall()
            
            return [
                (row[0], row[1], row[2], row[3])
                for row in rows
            ]
        except Exception as e:
            logger.warning("Failed to fetch unindexed messages: %s", e)
            return []
    
    def _count_unindexed(self) -> int:
        if not self.vec_available or not self.api_key:
            return 0
        
        conn = self._thread_safe_conn()
        if conn is None:
            return 0
        
        try:
            role_clause, role_params = self._role_filter_sql("m")
            min_len_clause, min_len_params = self._min_length_sql("m")
            vec_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
            ).fetchone()
            
            if not vec_exists:
                row = conn.execute(
                    f"SELECT COUNT(*) as cnt FROM messages m WHERE m.content IS NOT NULL AND m.content <> '' {min_len_clause} {role_clause}",
                    min_len_params + role_params
                ).fetchone()
            else:
                row = conn.execute(
                    f"""SELECT COUNT(*) as cnt FROM messages m
                       LEFT JOIN message_vec v ON m.id = v.message_id
                       WHERE v.message_id IS NULL
                         AND m.content IS NOT NULL AND m.content <> '' {min_len_clause} {role_clause}""",
                    min_len_params + role_params
                ).fetchone()
            
            return row[0] if row else 0
        except Exception:
            return 0
    
    def _get_last_message_time(self) -> float:
        conn = self._thread_safe_conn()
        if conn is None:
            return 0.0
        try:
            row = conn.execute(
                "SELECT MAX(timestamp) as ts FROM messages"
            ).fetchone()
            ts = row[0] if row else None
            if ts is None:
                return 0.0
            if isinstance(ts, (int, float)):
                return float(ts)
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(str(ts))
                return dt.timestamp()
            except Exception:
                return 0.0
        except Exception:
            return 0.0

    def _start_auto_index_daemon(self):
        def _daemon_loop():
            logger.info("Idle-index daemon: started (interval=%ds)",
                         self._idle_index_interval)
            check_interval = min(self._idle_index_interval, 30)
            while not self._stop_event.is_set():
                self._stop_event.wait(check_interval)
                if self._stop_event.is_set():
                    break
                try:
                    last_msg_time = self._get_last_message_time()
                    if last_msg_time <= 0:
                        continue
                    now = time.time()
                    idle_seconds = now - last_msg_time
                    if idle_seconds < self._idle_index_interval:
                        continue
                    if last_msg_time <= self._last_index_time:
                        continue
                    count = self._count_unindexed()
                    if count <= 0:
                        continue
                    logger.info("Idle-index daemon: idle %.0fs >= %ds, %d unindexed → starting batch",
                                 idle_seconds, self._idle_index_interval, count)
                    if not self._indexing_lock.acquire(blocking=False):
                        continue
                    try:
                        items = self._fetch_unindexed(limit=500)
                        if items:
                            before = self.indexer.indexed_count
                            before_failed = self.indexer.failed_count
                            self.indexer.index_batch(items, stop_event=self._stop_event)
                            batch_indexed = self.indexer.indexed_count - before
                            batch_failed = self.indexer.failed_count - before_failed
                            batch_skipped = len(items) - batch_indexed - batch_failed
                            if batch_indexed > 0:
                                self._last_index_time = last_msg_time
                            logger.info(
                                "Idle-index daemon: complete → total=%d, indexed=%d, failed=%d, skipped=%d",
                                len(items), batch_indexed, batch_failed, batch_skipped,
                            )
                    finally:
                        self._indexing_lock.release()
                except Exception as e:
                    logger.warning("Idle-index daemon: error → %s", e)
            logger.info("Idle-index daemon: stopped")
        
        thread = threading.Thread(target=_daemon_loop, daemon=True, name="hybrid-idle-index")
        thread.start()
    
    def stop_auto_index_daemon(self):
        self._stop_event.set()
    
    def _index_unindexed_batch(self, limit: int = None) -> Dict[str, Any]:
        if not self.vec_available or not self.api_key:
            return {"error": "hybrid search not available (sqlite-vec or API key missing)"}
        
        if not self._indexing_lock.acquire(blocking=False):
            return {"error": "indexing already in progress"}
        
        try:
            items = self._fetch_unindexed(limit=limit)
            if not items:
                logger.info("Index batch: no unindexed messages found")
                return {"indexed": 0, "message": "No unindexed messages found"}
            
            before_indexed = self.indexer.indexed_count
            before_failed = self.indexer.failed_count
            
            logger.info(
                "Index batch: starting → %d items to index (limit=%s)",
                len(items), limit or "none",
            )
            
            self.indexer.index_batch(items)
            
            indexed = self.indexer.indexed_count - before_indexed
            failed = self.indexer.failed_count - before_failed
            skipped = len(items) - indexed - failed
            
            logger.info(
                "Index batch: complete → total=%d, indexed=%d, failed=%d, skipped=%d",
                len(items), indexed, failed, skipped,
            )
            
            return {
                "indexed": indexed,
                "failed": failed,
                "total_processed": len(items),
            }
        finally:
            self._indexing_lock.release()
    
    def index_session(self, session_id: str = None):
        result = self._index_unindexed_batch()
        total = result.get("total_processed", 0)
        if total > 0:
            logger.info(
                "Session finalize: complete → total=%d, indexed=%d, failed=%d",
                total, result.get("indexed", 0), result.get("failed", 0),
            )
    
    def index_status(self) -> Dict[str, Any]:
        conn = self._thread_safe_conn()
        if conn is None:
            return {"error": "Cannot connect to database"}
        try:
            role_clause, role_params = self._role_filter_sql("m")
            min_len_clause, min_len_params = self._min_length_sql("m")
            
            total_all_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM messages"
            ).fetchone()
            total_all = total_all_row[0] if total_all_row else 0
            
            empty_content_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE content IS NULL OR content = ''"
            ).fetchone()
            empty_content = empty_content_row[0] if empty_content_row else 0
            
            excluded_by_role_row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM messages WHERE content IS NOT NULL AND content <> '' AND role NOT IN ({','.join('?' for _ in self.index_roles)})",
                list(self.index_roles)
            ).fetchone()
            excluded_by_role = excluded_by_role_row[0] if excluded_by_role_row else 0
            
            excluded_by_length = 0
            if self.min_content_length and self.min_content_length > 0:
                excluded_by_length_row = conn.execute(
                    f"SELECT COUNT(*) as cnt FROM messages WHERE content IS NOT NULL AND content <> '' AND LENGTH(content) < ? AND role IN ({','.join('?' for _ in self.index_roles)})",
                    [self.min_content_length] + list(self.index_roles)
                ).fetchone()
                excluded_by_length = excluded_by_length_row[0] if excluded_by_length_row else 0
            
            total_indexable_row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM messages m WHERE m.content IS NOT NULL AND m.content <> '' {min_len_clause} {role_clause}",
                min_len_params + role_params
            ).fetchone()
            total_indexable = total_indexable_row[0] if total_indexable_row else 0
            
            non_indexable = empty_content + excluded_by_role + excluded_by_length
            
            vec_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
            ).fetchone()
            
            indexed = 0
            if vec_exists:
                idx_row = conn.execute(
                    f"SELECT COUNT(*) as cnt FROM message_vec v JOIN messages m ON v.message_id = m.id {('WHERE ' + role_clause.replace('AND ', '')) if role_clause.strip() else ''}",
                    role_params
                ).fetchone()
                indexed = idx_row[0] if idx_row else 0
            
            truly_unindexed = max(0, total_indexable - indexed)
            
            role_breakdown = {}
            try:
                role_rows = conn.execute(
                    "SELECT role, COUNT(*) as cnt FROM messages WHERE content IS NOT NULL AND content <> '' GROUP BY role"
                ).fetchall()
                for row in role_rows:
                    role_breakdown[row[0]] = row[1]
            except Exception:
                pass
            
            indexed_by_role = {}
            if vec_exists:
                try:
                    idx_role_rows = conn.execute(
                        "SELECT m.role, COUNT(*) as cnt FROM message_vec v JOIN messages m ON v.message_id = m.id GROUP BY m.role"
                    ).fetchall()
                    for row in idx_role_rows:
                        indexed_by_role[row[0]] = row[1]
                except Exception:
                    pass
            
            unindexed_by_role = {}
            for role in self.index_roles:
                total_in_role = role_breakdown.get(role, 0)
                if self.min_content_length and self.min_content_length > 0:
                    try:
                        role_len_row = conn.execute(
                            "SELECT COUNT(*) as cnt FROM messages WHERE content IS NOT NULL AND content <> '' AND LENGTH(content) >= ? AND role = ?",
                            [self.min_content_length, role]
                        ).fetchone()
                        total_in_role = role_len_row[0] if role_len_row else 0
                    except Exception:
                        pass
                idx_in_role = indexed_by_role.get(role, 0)
                unindexed_in_role = total_in_role - idx_in_role
                if unindexed_in_role > 0:
                    unindexed_by_role[role] = unindexed_in_role
            
            return {
                "total_messages": total_all,
                "total_indexable": total_indexable,
                "indexed": indexed,
                "truly_unindexed": truly_unindexed,
                "non_indexable": non_indexable,
                "non_indexable_breakdown": {
                    "empty_content": empty_content,
                    "excluded_by_role": excluded_by_role,
                    "excluded_by_length": excluded_by_length,
                },
                "indexing_progress": f"{indexed}/{total_indexable}",
                "indexed_count": self.indexer.indexed_count,
                "failed_count": self.indexer.failed_count,
                "vec_available": self.vec_available,
                "index_roles": self.index_roles,
                "role_breakdown": role_breakdown,
                "indexed_by_role": indexed_by_role,
                "unindexed_by_role": unindexed_by_role,
                "min_content_length": self.min_content_length,
                "batch_token_limit": self.batch_token_limit,
                "idle_index_interval": self._idle_index_interval,
                "embedding_dim": self.embedding_dim,
            }
        except Exception as e:
            return {"error": str(e)}

    def rebuild_index(self) -> Dict[str, Any]:
        if not self.vec_available or not self.api_key:
            return {"error": "hybrid search not available (sqlite-vec or API key missing)"}
        
        if not self._indexing_lock.acquire(blocking=False):
            return {"error": "indexing already in progress, cannot rebuild now"}
        
        try:
            db_path = getattr(self.db, 'db_path', None) or getattr(self.db, '_db_path', None)
            if not db_path:
                try:
                    from hermes_constants import get_hermes_home
                    db_path = str(get_hermes_home() / "state.db")
                except Exception:
                    pass
            if not db_path:
                return {"error": "Cannot determine db path for rebuild"}
            
            write_conn = None
            try:
                from pysqlite3 import dbapi2 as pysqlite
                write_conn = pysqlite.connect(db_path, check_same_thread=False)
            except ImportError:
                import sqlite3
                write_conn = sqlite3.connect(db_path)
            
            try:
                write_conn.execute("DROP TABLE IF EXISTS message_vec")
                write_conn.commit()
                logger.info("Rebuild index: dropped message_vec table")
            finally:
                try:
                    write_conn.close()
                except Exception:
                    pass
            
            cached = getattr(_thread_local, 'hybrid_conn', None)
            if cached is not None:
                try:
                    cached.close()
                except Exception:
                    pass
                _thread_local.hybrid_conn = None
            
            items = self._fetch_unindexed()
            if not items:
                logger.info("Rebuild index: no messages to index after dropping message_vec")
                return {"indexed": 0, "message": "No messages to index after rebuild"}
            
            before_indexed = self.indexer.indexed_count
            before_failed = self.indexer.failed_count
            
            logger.info("Rebuild index: starting → %d items to re-index", len(items))
            
            self.indexer.index_batch(items)
            
            indexed = self.indexer.indexed_count - before_indexed
            failed = self.indexer.failed_count - before_failed
            skipped = len(items) - indexed - failed
            
            logger.info(
                "Rebuild index: complete → total=%d, indexed=%d, failed=%d, skipped=%d",
                len(items), indexed, failed, skipped,
            )
            
            return {
                "indexed": indexed,
                "failed": failed,
                "total_processed": len(items),
            }
        except Exception as e:
            logger.error("Rebuild index failed: %s", e, exc_info=True)
            return {"error": str(e)}
        finally:
            try:
                self._indexing_lock.release()
            except RuntimeError:
                pass


def get_hybrid_search(db, config: dict = None) -> "HybridSessionSearch":
    global _hybrid_instance
    with _hybrid_lock:
        if _hybrid_instance is not None:
            return _hybrid_instance
        instance = HybridSessionSearch(db, config=config or {})
        if instance.auto_index and instance.vec_available and instance.api_key:
            instance._start_auto_index_daemon()
        _hybrid_instance = instance
        return instance


def reset_hybrid_search():
    global _hybrid_instance
    with _hybrid_lock:
        if _hybrid_instance is not None:
            _hybrid_instance.stop_auto_index_daemon()
            _hybrid_instance = None


def check_hybrid_search_requirements() -> bool:
    """检查混合搜索是否可用。"""
    if not _check_sqlite_vec_available():
        logger.info("Hybrid search unavailable: sqlite-vec not installed")
        return False
    
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not api_key:
        try:
            from hermes_cli.config import load_config
            config = load_config()
            hybrid_config = config.get("auxiliary", {}).get("session_search", {}).get("hybrid", {})
            api_key = hybrid_config.get("embed_api_key", "")
        except Exception:
            pass
    
    if not api_key:
        logger.info("Hybrid search unavailable: no embed API key configured")
        return False
    
    return True
