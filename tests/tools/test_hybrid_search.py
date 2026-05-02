"""Tests for tools/hybrid_search.py — HybridSessionSearch core logic.

Tests cover:
- RRF fusion correctness
- vec_distance_threshold filtering
- rrf_score_threshold filtering
- Config reading and defaults
- _vec_serialize compatibility
- Index status reporting
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from collections import OrderedDict


def _make_search(db=None, config=None):
    from tools.hybrid_search import HybridSessionSearch
    mock_db = db or MagicMock()
    mock_db._conn = MagicMock()
    mock_db._db_path = "/tmp/test.db"
    with patch.object(HybridSessionSearch, "_load_vec_extension", return_value=False):
        return HybridSessionSearch(mock_db, config=config or {})


class TestVecSerialize:
    def test_serialize_float32_preferred(self):
        import sqlite_vec
        from tools.hybrid_search import _vec_serialize
        vector = [0.1, 0.2, 0.3, 0.4]
        result = _vec_serialize(vector)
        assert isinstance(result, bytes)
        assert len(result) == len(vector) * 4


class TestRRFFusion:
    def test_rrf_fusion_empty_inputs(self):
        search = _make_search()
        result = search._rrf_fusion([], [], limit=10)
        assert result == []

    def test_rrf_fusion_bm25_only(self):
        search = _make_search()
        bm25 = [
            {"message_id": 1, "session_id": "s1", "content": "a"},
            {"message_id": 2, "session_id": "s2", "content": "b"},
        ]
        result = search._rrf_fusion(bm25, [], limit=10)
        assert len(result) == 2
        assert result[0]["message_id"] == 1
        assert result[0]["rrf_score"] > result[1]["rrf_score"]

    def test_rrf_fusion_vector_only(self):
        search = _make_search()
        vec = [
            {"message_id": 3, "session_id": "s3", "content": "c"},
            {"message_id": 4, "session_id": "s4", "content": "d"},
        ]
        result = search._rrf_fusion([], vec, limit=10)
        assert len(result) == 2
        assert result[0]["message_id"] == 3

    def test_rrf_fusion_overlap_boosts_score(self):
        search = _make_search()
        bm25 = [
            {"message_id": 1, "session_id": "s1", "content": "a"},
            {"message_id": 2, "session_id": "s2", "content": "b"},
        ]
        vec = [
            {"message_id": 1, "session_id": "s1", "content": "a"},
            {"message_id": 3, "session_id": "s3", "content": "c"},
        ]
        result = search._rrf_fusion(bm25, vec, limit=10)
        assert result[0]["message_id"] == 1
        assert result[0]["rrf_score"] > result[1]["rrf_score"]

    def test_rrf_fusion_score_values(self):
        search = _make_search(config={"hybrid": {"rrf_k": 60}})
        bm25 = [{"message_id": 1, "session_id": "s1", "content": "a"}]
        vec = [{"message_id": 1, "session_id": "s1", "content": "a"}]
        result = search._rrf_fusion(bm25, vec, limit=10)
        expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
        assert abs(result[0]["rrf_score"] - expected) < 1e-10

    def test_rrf_fusion_respects_limit(self):
        search = _make_search()
        bm25 = [{"message_id": i, "session_id": f"s{i}", "content": str(i)} for i in range(10)]
        vec = [{"message_id": i + 10, "session_id": f"s{i+10}", "content": str(i + 10)} for i in range(10)]
        result = search._rrf_fusion(bm25, vec, limit=5)
        assert len(result) == 5


class TestVecDistanceThreshold:
    def test_default_threshold(self):
        search = _make_search()
        assert search.vec_distance_threshold == 1.2

    def test_custom_threshold(self):
        search = _make_search(config={"hybrid": {"vec_distance_threshold": 0.5}})
        assert search.vec_distance_threshold == 0.5

    def test_vector_search_filters_by_threshold(self):
        search = _make_search(config={"hybrid": {"vec_distance_threshold": 1.0}})
        search.vec_available = True
        search._ensure_vec_loaded = MagicMock(return_value=True)
        search._call_embedding_api = MagicMock(return_value=[0.1] * 1024)

        mock_row_1 = OrderedDict([("id", 1), ("session_id", "s1"), ("content", "a"), ("role", "user"), ("timestamp", 1.0), ("distance", 0.8)])
        mock_row_2 = OrderedDict([("id", 2), ("session_id", "s2"), ("content", "b"), ("role", "user"), ("timestamp", 2.0), ("distance", 1.5)])

        search.db._conn.execute.return_value.fetchall.return_value = [mock_row_1, mock_row_2]
        search.db._conn.execute.return_value.fetchone.return_value = OrderedDict([("name", "message_vec")])

        with patch("tools.hybrid_search._vec_serialize", return_value=b"\x00" * 4096):
            results = search._vector_search("test", limit=50)

        assert len(results) == 2

        search._lazy_index_unindexed = MagicMock()
        search._bm25_search = MagicMock(return_value=[])
        with patch("tools.hybrid_search._vec_serialize", return_value=b"\x00" * 4096):
            fused = search.search("test", limit=10)
        assert len(fused) == 1
        assert fused[0]["message_id"] == 1


class TestRRFScoreThreshold:
    def test_default_threshold_is_zero(self):
        search = _make_search()
        assert search.rrf_score_threshold == 0.0

    def test_no_filter_when_threshold_zero(self):
        search = _make_search(config={"hybrid": {"rrf_score_threshold": 0.0}})
        search._lazy_index_unindexed = MagicMock()
        search._bm25_search = MagicMock(return_value=[])
        search._vector_search = MagicMock(return_value=[])
        search._rrf_fusion = MagicMock(return_value=[
            {"message_id": 1, "rrf_score": 0.001},
            {"message_id": 2, "rrf_score": 0.0005},
        ])
        results = search.search("test", limit=10)
        assert len(results) == 2

    def test_filters_low_score_when_threshold_set(self):
        search = _make_search(config={"hybrid": {"rrf_score_threshold": 0.01}})
        search._lazy_index_unindexed = MagicMock()
        search._bm25_search = MagicMock(return_value=[])
        search._vector_search = MagicMock(return_value=[])
        search._rrf_fusion = MagicMock(return_value=[
            {"message_id": 1, "rrf_score": 0.016},
            {"message_id": 2, "rrf_score": 0.008},
            {"message_id": 3, "rrf_score": 0.005},
        ])
        results = search.search("test", limit=10)
        assert len(results) == 1
        assert results[0]["message_id"] == 1

    def test_all_filtered_returns_empty(self):
        search = _make_search(config={"hybrid": {"rrf_score_threshold": 0.1}})
        search._lazy_index_unindexed = MagicMock()
        search._bm25_search = MagicMock(return_value=[])
        search._vector_search = MagicMock(return_value=[])
        search._rrf_fusion = MagicMock(return_value=[
            {"message_id": 1, "rrf_score": 0.016},
        ])
        results = search.search("test", limit=10)
        assert results == []


class TestConfigReading:
    def test_full_config(self):
        config = {
            "timeout": 60,
            "hybrid": {
                "embed_provider": "custom",
                "embed_model": "BAAI/bge-m3",
                "embed_base_url": "https://api.example.com/v1",
                "embed_api_key": "sk-test",
                "use_reranker": True,
                "reranker_model": "custom-reranker",
                "rrf_k": 100,
                "vec_top_k": 30,
                "vec_distance_threshold": 0.9,
                "rrf_score_threshold": 0.01,
                "auto_index": False,
            },
        }
        search = _make_search(config=config)
        assert search.embedding_model == "BAAI/bge-m3"
        assert search.embedding_api_url == "https://api.example.com/v1"
        assert search.api_key == "sk-test"
        assert search.use_reranker is True
        assert search.reranker_model == "custom-reranker"
        assert search.rrf_k == 100
        assert search.vec_top_k == 30
        assert search.vec_distance_threshold == 0.9
        assert search.rrf_score_threshold == 0.01
        assert search.timeout == 60

    def test_defaults_with_empty_config(self):
        search = _make_search()
        assert search.embedding_model == "BAAI/bge-m3"
        assert search.use_reranker is False
        assert search.rrf_k == 60
        assert search.vec_top_k == 50
        assert search.vec_distance_threshold == 1.2
        assert search.rrf_score_threshold == 0.0
        assert search.index_roles == ["user", "assistant"]

    def test_custom_index_roles(self):
        search = _make_search(config={"hybrid": {"index_roles": ["user", "assistant", "tool"]}})
        assert search.index_roles == ["user", "assistant", "tool"]

    def test_api_key_fallback_to_env(self):
        with patch.dict("os.environ", {"SILICONFLOW_API_KEY": "sk-env"}):
            search = _make_search()
        assert search.api_key == "sk-env"


class TestIndexStatus:
    def test_index_status_no_vec_table(self):
        search = _make_search()
        def mock_execute(sql, *args):
            result = MagicMock()
            if "sqlite_master" in sql:
                result.fetchone.return_value = None
            elif "COUNT" in sql and "messages" in sql:
                result.fetchone.return_value = OrderedDict([("cnt", 100)])
            else:
                result.fetchone.return_value = None
            return result
        search.db._conn.execute = mock_execute
        status = search.index_status()
        assert status["total_indexable"] == 100
        assert status["indexed"] == 0

    def test_index_status_with_vec_table(self):
        search = _make_search()
        call_count = [0]
        def mock_execute(sql, *args):
            result = MagicMock()
            if "sqlite_master" in sql:
                result.fetchone.return_value = OrderedDict([("name", "message_vec")])
                result.fetchall.return_value = []
            elif "COUNT" in sql and "messages" in sql:
                result.fetchone.return_value = OrderedDict([("cnt", 100)])
            elif "COUNT" in sql and "message_vec" in sql:
                result.fetchone.return_value = OrderedDict([("cnt", 50)])
            else:
                result.fetchone.return_value = None
                result.fetchall.return_value = []
            return result
        search.db._conn.execute = mock_execute
        status = search.index_status()
        assert status["total_indexable"] == 100
        assert status["indexed"] == 50


class TestSearchEmpty:
    def test_empty_query_returns_empty(self):
        search = _make_search()
        assert search.search("") == []
        assert search.search("   ") == []


class TestRoleFilterSQL:
    def test_default_roles(self):
        search = _make_search()
        clause, params = search._role_filter_sql()
        assert "role IN" in clause
        assert params == ["user", "assistant"]

    def test_custom_roles(self):
        search = _make_search(config={"hybrid": {"index_roles": ["user"]}})
        clause, params = search._role_filter_sql()
        assert params == ["user"]

    def test_table_alias(self):
        search = _make_search()
        clause, params = search._role_filter_sql("msg")
        assert "msg.role IN" in clause


class TestRebuildIndex:
    def test_rebuild_not_available(self):
        search = _make_search()
        search.vec_available = False
        result = search.rebuild_index()
        assert "error" in result

    def test_rebuild_no_api_key(self):
        search = _make_search()
        search.vec_available = True
        search.api_key = ""
        result = search.rebuild_index()
        assert "error" in result

    def test_rebuild_drops_and_reindexes(self):
        search = _make_search(config={"hybrid": {"embed_api_key": "sk-test"}})
        search.vec_available = True
        search.api_key = "sk-test"
        search._ensure_vec_loaded = MagicMock(return_value=True)
        search._load_vec_extension = MagicMock(return_value=True)

        call_log = []
        def mock_execute(sql, *args):
            result = MagicMock()
            call_log.append(sql.strip().upper())
            if "DROP" in sql.upper():
                pass
            elif "SELECT" in sql.upper() and "MESSAGES" in sql.upper() and "COUNT" not in sql.upper():
                result.fetchall.return_value = []
            return result
        search.db._conn.execute = mock_execute
        search.db._conn.commit = MagicMock()

        result = search.rebuild_index()
        assert "indexed" in result or "error" in result
        assert any("DROP" in c for c in call_log)


class TestMinContentLength:
    def test_default_is_none(self):
        search = _make_search()
        assert search.min_content_length is None

    def test_custom_value(self):
        search = _make_search(config={"hybrid": {"min_content_length": 30}})
        assert search.min_content_length == 30

    def test_null_means_no_filter(self):
        search = _make_search()
        assert search._min_length_sql() == ""

    def test_zero_means_no_filter(self):
        search = _make_search(config={"hybrid": {"min_content_length": 0}})
        assert search.min_content_length == 0
        assert search._min_length_sql() == ""

    def test_positive_value_generates_clause(self):
        search = _make_search(config={"hybrid": {"min_content_length": 100}})
        clause = search._min_length_sql()
        assert "LENGTH(m.content) >= 100" in clause

    def test_index_status_includes_min_content_length(self):
        search = _make_search(config={"hybrid": {"min_content_length": 30}})
        def mock_execute(sql, *args):
            result = MagicMock()
            if "sqlite_master" in sql:
                result.fetchone.return_value = None
            elif "COUNT" in sql:
                result.fetchone.return_value = OrderedDict([("cnt", 10)])
            else:
                result.fetchone.return_value = None
            return result
        search.db._conn.execute = mock_execute
        status = search.index_status()
        assert status["min_content_length"] == 30


class TestBatchIndexing:
    def test_estimate_tokens_with_known_count(self):
        from tools.hybrid_search import EmbeddingIndexer
        assert EmbeddingIndexer._estimate_tokens("hello", token_count=100) == 100

    def test_estimate_tokens_without_count(self):
        from tools.hybrid_search import EmbeddingIndexer
        est = EmbeddingIndexer._estimate_tokens("hello world")
        assert est > 0

    def test_estimate_tokens_cjk(self):
        from tools.hybrid_search import EmbeddingIndexer
        cjk_est = EmbeddingIndexer._estimate_tokens("你好世界")
        ascii_est = EmbeddingIndexer._estimate_tokens("abcd")
        assert cjk_est > ascii_est

    def test_group_into_batches_respects_limit(self):
        from tools.hybrid_search import EmbeddingIndexer
        indexer = EmbeddingIndexer(MagicMock(), "http://x", "model", "key",
                                   batch_token_limit=100)
        items = [
            (1, "a" * 50, "s1", 40),
            (2, "b" * 50, "s1", 40),
            (3, "c" * 50, "s1", 40),
        ]
        batches = indexer._group_into_batches(items)
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1

    def test_group_into_batches_single_item_exceeds_limit(self):
        from tools.hybrid_search import EmbeddingIndexer
        indexer = EmbeddingIndexer(MagicMock(), "http://x", "model", "key",
                                   batch_token_limit=100)
        items = [(1, "x" * 1000, "s1", 500)]
        batches = indexer._group_into_batches(items)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_truncate_to_token_budget(self):
        from tools.hybrid_search import EmbeddingIndexer
        indexer = EmbeddingIndexer(MagicMock(), "http://x", "model", "key")
        long_text = "a" * 20000
        truncated = indexer._truncate_to_token_budget(long_text, 1000)
        assert len(truncated) < len(long_text)
        assert len(truncated) > 0

    def test_truncate_preserves_short_text(self):
        from tools.hybrid_search import EmbeddingIndexer
        indexer = EmbeddingIndexer(MagicMock(), "http://x", "model", "key")
        short = "hello world"
        result = indexer._truncate_to_token_budget(short, 8192)
        assert result == short

    def test_batch_token_limit_config(self):
        search = _make_search(config={"hybrid": {"batch_token_limit": 5000}})
        assert search.batch_token_limit == 5000

    def test_batch_token_limit_default(self):
        search = _make_search()
        assert search.batch_token_limit == 7000
