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
import queue
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _row_get(row, key, default=None):
    try:
        return row[key]
    except (TypeError, KeyError, IndexError):
        if isinstance(row, (tuple, list)):
            col_map = {
                "id": 0, "message_id": 0, "session_id": 1,
                "content": 2, "role": 3, "timestamp": 4,
                "distance": 5, "file": 2, "name": 0, "cnt": 0,
            }
            idx = col_map.get(key)
            if idx is not None and idx < len(row):
                return row[idx]
        return default


def _vec_serialize(vector: List[float]) -> bytes:
    """Serialize float vector for sqlite-vec. Handles API name differences."""
    import sqlite_vec
    fn = getattr(sqlite_vec, 'serialize_float32', None) or sqlite_vec.serialize_f32
    return fn(vector)


def _check_sqlite_vec_available() -> bool:
    """Check if sqlite-vec extension is available."""
    try:
        import sqlite_vec
        return True
    except ImportError:
        return False


class EmbeddingIndexer:
    """异步 embedding 索引器 - 队列 + 批量消费者模式。"""
    
    _MODEL_MAX_TOKENS = 8192
    
    def __init__(self, db, api_url: str, model: str, api_key: str,
                 min_content_length: int = None, batch_token_limit: int = 7000):
        self.db = db
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.min_content_length = min_content_length
        self.batch_token_limit = batch_token_limit
        
        self.queue = queue.Queue(maxsize=1000)
        
        self.indexed_count = 0
        self.failed_count = 0
        
        self._start_worker()
    
    def _start_worker(self):
        def _worker():
            while True:
                item = self.queue.get()
                if item is None:
                    break
                
                message_id, content, session_id, token_count = item
                try:
                    self._index_one(message_id, content, session_id, token_count)
                    self.indexed_count += 1
                except Exception as e:
                    logger.warning("Failed to index message %d: %s", message_id, e)
                    self.failed_count += 1
                
                self.queue.task_done()
        
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
    
    def enqueue(self, message_id: int, content: str, session_id: str, token_count: int = None):
        if self.min_content_length is not None and self.min_content_length > 0 and len(content) < self.min_content_length:
            return
        
        try:
            self.queue.put_nowait((message_id, content, session_id, token_count))
        except queue.Full:
            logger.warning("Index queue full, dropping message %d", message_id)
    
    @staticmethod
    def _estimate_tokens(content: str, token_count: int = None) -> int:
        if token_count is not None and token_count > 0:
            return token_count
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
    
    def _index_one(self, message_id: int, content: str, session_id: str, token_count: int = None):
        try:
            import sqlite_vec
            
            estimated_tokens = self._estimate_tokens(content, token_count)
            
            if estimated_tokens > self._MODEL_MAX_TOKENS:
                content = self._truncate_to_token_budget(content, int(self._MODEL_MAX_TOKENS * 0.9))
                if not content or not content.strip():
                    return
            
            embedding = self._call_embedding_api_with_retry(content)
            if not embedding:
                return
            
            db_path = getattr(self.db, 'db_path', None) or getattr(self.db, '_db_path', None)
            if not db_path:
                try:
                    row = self.db._conn.execute("PRAGMA database_list").fetchone()
                    if row:
                        db_path = row["file"] if hasattr(row, "keys") else row[2]
                except Exception:
                    pass
            
            if not db_path or db_path == "":
                logger.warning("Cannot determine db path for background indexing")
                return
            
            try:
                from pysqlite3 import dbapi2 as pysqlite
                conn = pysqlite.connect(db_path)
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
                    conn.load_extension(str(vec_path))
                    try:
                        conn.enable_load_extension(False)
                    except AttributeError:
                        pass
            
            vec_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
            ).fetchone()
            
            if not vec_exists:
                conn.execute(
                    """CREATE VIRTUAL TABLE IF NOT EXISTS message_vec USING vec0(
                        message_id INTEGER PRIMARY KEY,
                        embedding FLOAT[1024],
                        session_id TEXT
                    )"""
                )
                conn.commit()
            
            conn.execute(
                """INSERT OR REPLACE INTO message_vec 
                   (message_id, embedding, session_id) 
                   VALUES (?, ?, ?)""",
                (message_id, _vec_serialize(embedding), session_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to index message %d: %s", message_id, e)
    
    def index_batch(self, items: list):
        """批量索引：按 token 分组，每组一次 API 调用 + 一次 DB 写入。
        
        items: [(message_id, content, session_id, token_count), ...]
        """
        if not items or not self.api_key:
            return
        
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
            conn = pysqlite.connect(db_path)
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
                    """CREATE VIRTUAL TABLE IF NOT EXISTS message_vec USING vec0(
                        message_id INTEGER PRIMARY KEY,
                        embedding FLOAT[1024],
                        session_id TEXT
                    )"""
                )
                conn.commit()
            
            batches = self._group_into_batches(items)
            
            for batch in batches:
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
                    self.failed_count += len(valid_items)
                    continue
                
                for (msg_id, sid), emb in zip(valid_items, embeddings):
                    try:
                        conn.execute(
                            """INSERT OR REPLACE INTO message_vec 
                               (message_id, embedding, session_id) 
                               VALUES (?, ?, ?)""",
                            (msg_id, _vec_serialize(emb), sid)
                        )
                        self.indexed_count += 1
                    except Exception as e:
                        logger.warning("Failed to insert embedding for %d: %s", msg_id, e)
                        self.failed_count += 1
                
                conn.commit()
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
        """批量调用 embedding API。"""
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
                data = response.json()["data"]
                data.sort(key=lambda x: x["index"])
                return [d["embedding"] for d in data]
            
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
        """批量失败时回退到逐条调用。"""
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
                return response.json()["data"][0]["embedding"]
            
            except httpx.TimeoutException:
                logger.warning("Embedding API timeout (attempt %d/%d)", attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            
            except Exception as e:
                logger.error("Embedding API error: %s", e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def wait_for_completion(self, timeout: int = 300):
        """等待队列处理完成。"""
        try:
            self.queue.join()
        except Exception:
            pass
    
    def shutdown(self):
        """关闭索引器。"""
        self.queue.put(None)  # 发送停止信号


class HybridSessionSearch:
    """混合搜索：BM25 + Vector + RRF。"""
    
    def __init__(self, db, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        
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
        _raw_min = hybrid_config.get("min_content_length")
        self.min_content_length = int(_raw_min) if _raw_min is not None else None
        self.batch_token_limit = hybrid_config.get("batch_token_limit", 7000)
        self.auto_index_threshold = hybrid_config.get("auto_index_threshold", 10)
        self.auto_index = hybrid_config.get("auto_index", True)
        self._indexing_lock = threading.Lock()
        self._auto_index_interval = hybrid_config.get("auto_index_interval", 30)
        self._stop_event = threading.Event()
        
        # 检查 sqlite-vec 可用性并加载扩展
        self.vec_available = self._load_vec_extension()
        if not self.vec_available:
            logger.warning("sqlite-vec not available, vector search disabled")
        
        # 初始化索引器
        self.indexer = EmbeddingIndexer(
            db, self.embedding_api_url, self.embedding_model, self.api_key,
            min_content_length=self.min_content_length,
            batch_token_limit=self.batch_token_limit,
        )
        
        # 启动后台自动索引守护线程
        if self.auto_index and self.vec_available and self.api_key:
            self._start_auto_index_daemon()

    def _role_filter_sql(self, table_alias: str = "m") -> tuple:
        return (
            f"AND {table_alias}.role IN ({','.join('?' for _ in self.index_roles)}) ",
            list(self.index_roles),
        )

    def _min_length_sql(self, table_alias: str = "m") -> str:
        if self.min_content_length is not None and self.min_content_length > 0:
            return f"AND LENGTH({table_alias}.content) >= {self.min_content_length} "
        return ""
    
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
            try:
                from pysqlite3 import dbapi2 as pysqlite
                db_path = None
                try:
                    row = conn.execute("PRAGMA database_list").fetchone()
                    if row:
                        db_path = row["file"] if hasattr(row, "keys") else row[2]
                except Exception:
                    pass
                
                new_conn = pysqlite.connect(db_path or ":memory:")
                new_conn.enable_load_extension(True)
                sqlite_vec.load(new_conn)
                new_conn.enable_load_extension(False)
                new_conn.row_factory = pysqlite.Row
                
                self.db._conn = new_conn
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
        try:
            # 使用现有的 FTS5 搜索
            results = self.db.search_messages(
                query,
                limit=limit,
                offset=0,
            )
            
            # 转换为标准格式
            formatted = []
            for r in results:
                formatted.append({
                    "message_id": r.get("id"),
                    "session_id": r.get("session_id"),
                    "content": r.get("content", ""),
                    "role": r.get("role", ""),
                    "timestamp": r.get("timestamp", 0),
                    "snippet": r.get("snippet", ""),
                    "bm25_rank": r.get("rank", 0),
                })
            
            return formatted
        except Exception as e:
            logger.warning("BM25 search failed: %s", e, exc_info=True)
            return []
    
    def _ensure_vec_loaded(self):
        """确保主连接已加载 sqlite-vec 扩展。"""
        try:
            self.db._conn.execute("SELECT vec_version()").fetchone()
            return True
        except Exception:
            pass
        try:
            import sqlite_vec
            try:
                self.db._conn.enable_load_extension(True)
            except AttributeError:
                pass
            try:
                sqlite_vec.load(self.db._conn)
                try:
                    self.db._conn.enable_load_extension(False)
                except AttributeError:
                    pass
                return True
            except Exception:
                pass
            try:
                vec_path = sqlite_vec.loadable_path()
                if callable(vec_path):
                    vec_path = vec_path()
                self.db._conn.load_extension(str(vec_path))
                try:
                    self.db._conn.enable_load_extension(False)
                except AttributeError:
                    pass
                return True
            except Exception:
                pass
        except ImportError:
            pass
        return False

    def _vector_search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """向量语义搜索（使用 sqlite-vec）。"""
        if not self.vec_available:
            return []
        
        if not self._ensure_vec_loaded():
            return []
        
        query_embedding = self._call_embedding_api(query)
        if not query_embedding:
            return []
        
        try:
            import sqlite_vec
            
            vec_exists = self.db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
            ).fetchone()
            
            if not vec_exists:
                return []
            
            top_k = min(self.vec_top_k, limit)
            
            results = self.db._conn.execute(
                """SELECT m.id, m.session_id, m.content, m.role, m.timestamp,
                          distance
                   FROM message_vec
                   JOIN messages m ON m.id = message_vec.message_id
                   WHERE embedding MATCH ?
                     AND k = ?
                   ORDER BY distance""",
                (_vec_serialize(query_embedding), top_k)
            ).fetchall()
            
            formatted = []
            for r in results:
                distance = _row_get(r, "distance")
                if distance is None:
                    continue
                content = _row_get(r, "content") or ""
                formatted.append({
                    "message_id": _row_get(r, "id"),
                    "session_id": _row_get(r, "session_id"),
                    "content": content,
                    "role": _row_get(r, "role"),
                    "timestamp": _row_get(r, "timestamp"),
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
        """RRF 融合 - 合并 BM25 和 Vector 结果。"""
        from collections import defaultdict
        
        scores = defaultdict(float)
        result_map = {}
        
        # BM25 排名贡献
        for rank, result in enumerate(bm25_results, 1):
            msg_id = result["message_id"]
            scores[msg_id] += 1.0 / (self.rrf_k + rank)
            result_map[msg_id] = result
        
        # Vector 排名贡献
        for rank, result in enumerate(vector_results, 1):
            msg_id = result["message_id"]
            scores[msg_id] += 1.0 / (self.rrf_k + rank)
            result_map[msg_id] = result
        
        # 按融合分数排序
        fused = []
        for msg_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]:
            result = result_map[msg_id].copy()
            result["rrf_score"] = score
            fused.append(result)
        
        return fused
    
    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        
        query = query.strip()
        
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
                for c in candidates[:20]  # 只 rerank top 20
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
                result = candidates[idx].copy()
                result["rerank_score"] = hit["relevance_score"]
                reranked.append(result)
            
            return reranked[:limit]
        except Exception as e:
            logger.warning("Reranker failed: %s", e, exc_info=True)
            return candidates[:limit]
    
    def index_message_embedding(self, message_id: int, content: str, session_id: str):
        """将消息加入索引队列（异步）。"""
        self.indexer.enqueue(message_id, content, session_id)
    
    def _fetch_unindexed(self, session_id: str = None, limit: int = None) -> list:
        if not self.vec_available or not self.api_key:
            return []
        
        self._ensure_vec_loaded()
        role_clause, role_params = self._role_filter_sql()
        min_len_clause = self._min_length_sql()
        session_filter = "AND m.session_id = ? " if session_id else ""
        session_params = [session_id] if session_id else []
        limit_clause = f"LIMIT ? " if limit else ""
        limit_params = [limit] if limit else []
        
        try:
            vec_exists = self.db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
            ).fetchone()
            
            if not vec_exists:
                rows = self.db._conn.execute(
                    f"""SELECT m.id, m.content, m.session_id, m.token_count FROM messages m
                       WHERE m.content IS NOT NULL {min_len_clause}
                       {session_filter}
                       {role_clause}
                       ORDER BY m.timestamp DESC {limit_clause}""",
                    session_params + role_params + limit_params
                ).fetchall()
            else:
                rows = self.db._conn.execute(
                    f"""SELECT m.id, m.content, m.session_id, m.token_count FROM messages m
                       LEFT JOIN message_vec v ON m.id = v.message_id
                       WHERE v.message_id IS NULL
                         AND m.content IS NOT NULL {min_len_clause}
                       {session_filter}
                       {role_clause}
                       ORDER BY m.timestamp DESC {limit_clause}""",
                    session_params + role_params + limit_params
                ).fetchall()
            
            return [(
                _row_get(row, "id"),
                _row_get(row, "content"),
                _row_get(row, "session_id"),
                _row_get(row, "token_count"),
            ) for row in rows]
        except Exception as e:
            logger.warning("Failed to fetch unindexed messages: %s", e)
            return []
    
    def _count_unindexed(self) -> int:
        if not self.vec_available or not self.api_key:
            return 0
        
        try:
            role_clause, role_params = self._role_filter_sql("")
            min_len_clause = self._min_length_sql("")
            vec_exists = self.db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
            ).fetchone()
            
            if not vec_exists:
                row = self.db._conn.execute(
                    f"SELECT COUNT(*) as cnt FROM messages WHERE content IS NOT NULL {min_len_clause} {role_clause}",
                    role_params
                ).fetchone()
            else:
                row = self.db._conn.execute(
                    f"""SELECT COUNT(*) as cnt FROM messages m
                       LEFT JOIN message_vec v ON m.id = v.message_id
                       WHERE v.message_id IS NULL
                         AND m.content IS NOT NULL {min_len_clause} {role_clause}""",
                    role_params
                ).fetchone()
            
            return _row_get(row, "cnt", 0)
        except Exception:
            return 0
    
    def _start_auto_index_daemon(self):
        def _daemon_loop():
            logger.info("Auto-index daemon started (interval=%ds, threshold=%d)",
                         self._auto_index_interval, self.auto_index_threshold)
            while not self._stop_event.is_set():
                self._stop_event.wait(self._auto_index_interval)
                if self._stop_event.is_set():
                    break
                try:
                    if self._indexing_lock.locked():
                        continue
                    count = self._count_unindexed()
                    if count < self.auto_index_threshold:
                        continue
                    logger.info("Auto-index daemon: %d unindexed >= threshold %d, starting batch",
                                 count, self.auto_index_threshold)
                    if not self._indexing_lock.acquire(blocking=False):
                        continue
                    try:
                        items = self._fetch_unindexed(limit=500)
                        if items:
                            self.indexer.index_batch(items)
                    finally:
                        self._indexing_lock.release()
                except Exception as e:
                    logger.warning("Auto-index daemon error: %s", e)
            logger.info("Auto-index daemon stopped")
        
        thread = threading.Thread(target=_daemon_loop, daemon=True, name="hybrid-auto-index")
        thread.start()
    
    def stop_auto_index_daemon(self):
        self._stop_event.set()
    
    def _index_unindexed_batch(self, session_id: str = None, limit: int = None) -> Dict[str, Any]:
        if not self.vec_available or not self.api_key:
            return {"error": "hybrid search not available (sqlite-vec or API key missing)"}
        
        if self._indexing_lock.locked():
            return {"error": "indexing already in progress"}
        
        with self._indexing_lock:
            items = self._fetch_unindexed(session_id=session_id, limit=limit)
            if not items:
                return {"indexed": 0, "message": "No unindexed messages found"}
            
            before_indexed = self.indexer.indexed_count
            before_failed = self.indexer.failed_count
            
            self.indexer.index_batch(items)
            
            return {
                "indexed": self.indexer.indexed_count - before_indexed,
                "failed": self.indexer.failed_count - before_failed,
                "total_processed": len(items),
            }
    
    def index_session(self, session_id: str):
        if not session_id:
            return
        result = self._index_unindexed_batch(session_id=session_id)
        if result.get("total_processed", 0) > 0:
            logger.info("Indexed session %s: %s", session_id, result)
    
    def index_status(self) -> Dict[str, Any]:
        try:
            role_clause, role_params = self._role_filter_sql("")
            total = self.db._conn.execute(
                f"SELECT COUNT(*) as cnt FROM messages WHERE content IS NOT NULL {self._min_length_sql('')} {role_clause}",
                role_params
            ).fetchone()["cnt"]
            
            vec_exists = self.db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
            ).fetchone()
            
            if vec_exists:
                indexed = self.db._conn.execute(
                    "SELECT COUNT(*) as cnt FROM message_vec"
                ).fetchone()["cnt"]
            else:
                indexed = 0
            
            return {
                "total_indexable": total,
                "indexed": indexed,
                "unindexed": total - indexed,
                "indexing_progress": f"{indexed}/{total}",
                "queue_size": self.indexer.queue.qsize(),
                "indexed_count": self.indexer.indexed_count,
                "failed_count": self.indexer.failed_count,
                "vec_available": self.vec_available,
                "index_roles": self.index_roles,
                "min_content_length": self.min_content_length,
                "batch_token_limit": self.batch_token_limit,
                "auto_index_threshold": self.auto_index_threshold,
                "auto_index_interval": self._auto_index_interval,
            }
        except Exception as e:
            return {"error": str(e)}

    def rebuild_index(self) -> Dict[str, Any]:
        if not self.vec_available or not self.api_key:
            return {"error": "hybrid search not available (sqlite-vec or API key missing)"}
        
        self._ensure_vec_loaded()
        
        try:
            self.db._conn.execute("DROP TABLE IF EXISTS message_vec")
            self.db._conn.commit()
            logger.info("Dropped message_vec table for rebuild")
            
            self._load_vec_extension()
            
            return self._index_unindexed_batch()
        except Exception as e:
            logger.error("Rebuild index failed: %s", e, exc_info=True)
            return {"error": str(e)}


def check_hybrid_search_requirements() -> bool:
    """检查混合搜索是否可用。"""
    if not _check_sqlite_vec_available():
        logger.info("hybrid search unavailable: sqlite-vec not installed")
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
        logger.info("hybrid search unavailable: no embed API key configured")
        return False
    
    return True
