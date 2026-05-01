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
    """异步 embedding 索引器 - 队列 + 单消费者模式。"""
    
    def __init__(self, db, api_url: str, model: str, api_key: str):
        self.db = db
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        
        # 队列（最多 1000 条）
        self.queue = queue.Queue(maxsize=1000)
        
        # 统计信息
        self.indexed_count = 0
        self.failed_count = 0
        
        # 启动单个工作线程
        self._start_worker()
    
    def _start_worker(self):
        """启动消费者线程。"""
        def _worker():
            while True:
                item = self.queue.get()
                if item is None:  # 停止信号
                    break
                
                message_id, content, session_id = item
                try:
                    self._index_one(message_id, content, session_id)
                    self.indexed_count += 1
                except Exception as e:
                    logger.warning("Failed to index message %d: %s", message_id, e)
                    self.failed_count += 1
                
                self.queue.task_done()
        
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
    
    def enqueue(self, message_id: int, content: str, session_id: str):
        """将消息加入索引队列。"""
        # 过滤太短的消息
        if len(content) < 50:
            return
        
        try:
            self.queue.put_nowait((message_id, content, session_id))
        except queue.Full:
            logger.warning("Index queue full, dropping message %d", message_id)
    
    def _index_one(self, message_id: int, content: str, session_id: str):
        """索引单条消息（在后台线程中执行，使用独立连接）。"""
        try:
            import sqlite_vec
            
            embedding = self._call_embedding_api_with_retry(content)
            if not embedding:
                return
            
            db_path = getattr(self.db, '_db_path', None)
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
    
    def _call_embedding_api_with_retry(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """带重试的 embedding API 调用。"""
        if not self.api_key:
            return None
        
        import httpx
        
        for attempt in range(max_retries):
            try:
                response = httpx.post(
                    f"{self.api_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "input": text[:8000],  # 截断避免 token 限制
                    },
                    timeout=30
                )
                
                # 处理限流
                if response.status_code == 429:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.warning("Embedding API rate limited, waiting %ds", wait_time)
                    time.sleep(wait_time)
                    continue
                
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
        
        # 检查 sqlite-vec 可用性并加载扩展
        self.vec_available = self._load_vec_extension()
        if not self.vec_available:
            logger.warning("sqlite-vec not available, vector search disabled")
        
        # 初始化索引器
        self.indexer = EmbeddingIndexer(
            db, self.embedding_api_url, self.embedding_model, self.api_key
        )

    def _role_filter_sql(self, table_alias: str = "m") -> tuple:
        return (
            f"AND {table_alias}.role IN ({','.join('?' for _ in self.index_roles)}) ",
            list(self.index_roles),
        )
    
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
                new_conn.row_factory = conn.row_factory
                
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
                distance = r["distance"]
                if distance > self.vec_distance_threshold:
                    continue
                formatted.append({
                    "message_id": r["id"],
                    "session_id": r["session_id"],
                    "content": r["content"] or "",
                    "role": r["role"],
                    "timestamp": r["timestamp"],
                    "snippet": (r["content"] or "")[:200] + "...",
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
        
        self._lazy_index_unindexed()
        
        bm25_results = self._bm25_search(query, limit=50)
        vector_results = self._vector_search(query, limit=50)
        
        fused_results = self._rrf_fusion(bm25_results, vector_results, limit=limit * 2)
        
        if self.rrf_score_threshold > 0 and fused_results:
            fused_results = [r for r in fused_results if r.get("rrf_score", 0) >= self.rrf_score_threshold]
        
        if self.use_reranker and fused_results:
            fused_results = self._rerank_results(query, fused_results, limit=limit)
        
        for r in fused_results[:limit]:
            r["_diagnostics"] = {
                "bm25_hits": len(bm25_results),
                "vector_hits": len(vector_results),
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
    
    def _lazy_index_unindexed(self, max_messages: int = 500):
        if not self.vec_available or not self.api_key:
            return
        
        self._ensure_vec_loaded()
        role_clause, role_params = self._role_filter_sql()
        
        try:
            vec_exists = self.db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
            ).fetchone()
            
            if not vec_exists:
                unindexed = self.db._conn.execute(
                    f"""SELECT m.id, m.content, m.session_id FROM messages m
                       WHERE m.content IS NOT NULL AND LENGTH(m.content) >= 50
                       {role_clause}
                       ORDER BY m.timestamp DESC LIMIT ?""",
                    role_params + [max_messages]
                ).fetchall()
            else:
                unindexed = self.db._conn.execute(
                    f"""SELECT m.id, m.content, m.session_id FROM messages m
                       LEFT JOIN message_vec v ON m.id = v.message_id
                       WHERE v.message_id IS NULL
                         AND m.content IS NOT NULL AND LENGTH(m.content) >= 50
                         {role_clause}
                       ORDER BY m.timestamp DESC LIMIT ?""",
                    role_params + [max_messages]
                ).fetchall()
            
            if not unindexed:
                return
            
            count = 0
            for row in unindexed:
                self.indexer.enqueue(row["id"], row["content"], row["session_id"])
                count += 1
            
            if count > 0:
                logger.info("Lazy-indexing %d unindexed messages", count)
                self.indexer.wait_for_completion(timeout=300)
        except Exception as e:
            logger.warning("Lazy indexing failed: %s", e, exc_info=True)
    
    def index_session(self, session_id: str):
        if not self.vec_available or not self.api_key:
            return
        
        role_clause, role_params = self._role_filter_sql()
        
        try:
            vec_exists = self.db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
            ).fetchone()
            
            if not vec_exists:
                unindexed = self.db._conn.execute(
                    f"""SELECT m.id, m.content, m.session_id FROM messages m
                       WHERE m.session_id = ?
                         AND m.content IS NOT NULL AND LENGTH(m.content) >= 50
                         {role_clause}""",
                    [session_id] + role_params
                ).fetchall()
            else:
                unindexed = self.db._conn.execute(
                    f"""SELECT m.id, m.content, m.session_id FROM messages m
                       LEFT JOIN message_vec v ON m.id = v.message_id
                       WHERE m.session_id = ?
                         AND v.message_id IS NULL
                         AND m.content IS NOT NULL AND LENGTH(m.content) >= 50
                         {role_clause}""",
                    [session_id] + role_params
                ).fetchall()
            
            if not unindexed:
                return
            
            count = 0
            for row in unindexed:
                self.indexer.enqueue(row["id"], row["content"], row["session_id"])
                count += 1
            
            if count > 0:
                logger.info("Indexing %d messages for session %s", count, session_id)
        except Exception as e:
            logger.warning("Session indexing failed for %s: %s", session_id, e, exc_info=True)
    
    def index_status(self) -> Dict[str, Any]:
        try:
            role_clause, role_params = self._role_filter_sql("")
            total = self.db._conn.execute(
                f"SELECT COUNT(*) as cnt FROM messages WHERE content IS NOT NULL AND LENGTH(content) >= 50 {role_clause}",
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
            }
        except Exception as e:
            return {"error": str(e)}

    def rebuild_index(self) -> Dict[str, Any]:
        if not self.vec_available or not self.api_key:
            return {"error": "hybrid search not available (sqlite-vec or API key missing)"}
        
        self._ensure_vec_loaded()
        role_clause, role_params = self._role_filter_sql()
        
        try:
            self.db._conn.execute("DROP TABLE IF EXISTS message_vec")
            self.db._conn.commit()
            logger.info("Dropped message_vec table for rebuild")
            
            self._load_vec_extension()
            
            unindexed = self.db._conn.execute(
                f"""SELECT m.id, m.content, m.session_id FROM messages m
                   WHERE m.content IS NOT NULL AND LENGTH(m.content) >= 50
                   {role_clause}
                   ORDER BY m.timestamp DESC""",
                role_params
            ).fetchall()
            
            if not unindexed:
                return {"rebuilt": True, "indexed": 0, "message": "No messages to index"}
            
            for row in unindexed:
                self.indexer.enqueue(row["id"], row["content"], row["session_id"])
            
            logger.info("Rebuilding hybrid index: %d messages", len(unindexed))
            self.indexer.wait_for_completion(timeout=600)
            
            return {
                "rebuilt": True,
                "total_messages": len(unindexed),
                "indexed": self.indexer.indexed_count,
                "failed": self.indexer.failed_count,
                "index_roles": self.index_roles,
            }
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
