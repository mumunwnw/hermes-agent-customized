# Hybrid Session Search

## 架构概览

```
                          config.yaml
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    auxiliary.session_search   │   auxiliary.session_search.hybrid
    (LLM 摘要配置)             │   (Embedding/Reranker 配置)
    provider/model/base_url    │   embed_*/reranker_*/rrf_k
              │               │               │
              ▼               │               ▼
     auxiliary_client.py      │        hybrid_search.py
     async_call_llm()         │        HybridSessionSearch
     /chat/completions        │        /embeddings + /rerank
                              │
              ┌───────────────┘
              │
              ▼
     auxiliary.session_search.engine
     (bm25 | hybrid | auto)
              │
              ▼
     session_search_tool.py  ←── 路由入口
              │
    ┌─────────┼─────────┐
    │         │         │
  bm25     hybrid     auto
    │         │         │
    ▼         ▼         ▼
  FTS5    BM25+Vec   尝试hybrid
  搜索    +RRF融合   失败回退bm25
            │
            ▼
       LLM 摘要（auxiliary_client）
```

## 数据流

```
用户搜索 "部署数据库的问题"
        │
        ▼
session_search_tool.py
  │
  ├─ 读取 auxiliary.session_search.engine
  │
  ├─ engine == "bm25" → FTS5 搜索 + LLM 摘要
  │
  ├─ engine == "hybrid" →
  │   1. 读取 auxiliary.session_search 完整配置
  │   2. HybridSessionSearch(db, config=config)
  │      └─ _load_vec_extension() 加载 sqlite-vec
  │   3. _lazy_index_unindexed() 懒索引未索引消息
  │   4. search(query)
  │      ├─ BM25: db.search_messages() (FTS5)
  │      ├─ Vector: sqlite-vec KNN (vec_top_k + vec_distance_threshold)
  │      ├─ RRF: 融合排序
  │      ├─ RRF score threshold 过滤（可选，默认不过滤）
  │      └─ Reranker: 可选重排序
  │   5. LLM 摘要（auxiliary_client，配置独立）
  │
  └─ engine == "auto" → 尝试 hybrid，失败回退 bm25
```

## 索引机制

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: 搜索时懒索引（Lazy Indexing）                       │
│  触发：search() 调用时自动触发                                 │
│  逻辑：LEFT JOIN message_vec 找未索引消息 → 异步入队 → 等待完成 │
│  优点：零配置，搜索即索引，不遗漏                              │
│  缺点：首次搜索有延迟                                        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: 会话结束批量索引（Batch Indexing）                   │
│  触发：on_session_finalize 钩子（CLI退出、/new、/reset等）     │
│  实现：plugins/hybrid-search-indexer/ 插件                    │
│  逻辑：index_session(session_id) → 异步入队 → 后台线程处理     │
│  优点：下次搜索无需等待                                      │
│  缺点：需启用插件                                            │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: 手动批量索引（Manual Batch Indexing）               │
│  触发：scripts/index_embeddings.py                            │
│  用途：首次启用 hybrid 时的全量索引                            │
└─────────────────────────────────────────────────────────────┘
```

## 配置结构

```yaml
auxiliary:
  session_search:
    # LLM 摘要（由 auxiliary_client.py 读取）
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    extra_body: {}
    max_concurrency: 3

    # 搜索引擎
    engine: "bm25"            # "bm25" | "hybrid" | "auto"

    # Hybrid 专属
    hybrid:
      embed_provider: "custom"
      embed_model: "BAAI/bge-m3"
      embed_base_url: "https://api.siliconflow.cn/v1"
      embed_api_key: ""          # 回退 SILICONFLOW_API_KEY
      use_reranker: false
      reranker_provider: "custom"
      reranker_model: "BAAI/bge-reranker-v2-m3"
      reranker_base_url: ""
      reranker_api_key: ""
      rrf_k: 60
      vec_top_k: 50
      vec_distance_threshold: 1.2
      rrf_score_threshold: 0.0
      index_roles: ["user", "assistant"]
      min_content_length: null
      batch_token_limit: 7000
      auto_index_threshold: 10
      auto_index_interval: 30
      auto_index: true
```

### 配置字段职责

| 字段 | 读取者 | 含义 | API |
|------|--------|------|-----|
| `provider` | `auxiliary_client.py` | LLM 提供商 | - |
| `model` | `auxiliary_client.py` | LLM 摘要模型 | `/chat/completions` |
| `base_url` | `auxiliary_client.py` | LLM 端点 | - |
| `api_key` | `auxiliary_client.py` | LLM 密钥 | - |
| `engine` | `session_search_tool.py` | 搜索引擎类型 | - |
| `hybrid.embed_provider` | `hybrid_search.py` | Embedding 提供商 | - |
| `hybrid.embed_model` | `hybrid_search.py` | Embedding 模型 | `/embeddings` |
| `hybrid.embed_base_url` | `hybrid_search.py` | Embedding 端点 | - |
| `hybrid.embed_api_key` | `hybrid_search.py` | Embedding 密钥 | - |
| `hybrid.reranker_provider` | `hybrid_search.py` | Reranker 提供商 | - |
| `hybrid.reranker_model` | `hybrid_search.py` | Reranker 模型 | `/rerank` |
| `hybrid.reranker_base_url` | `hybrid_search.py` | Reranker 端点 | - |
| `hybrid.reranker_api_key` | `hybrid_search.py` | Reranker 密钥 | - |
| `hybrid.vec_top_k` | `hybrid_search.py` | 向量搜索返回条数 | - |
| `hybrid.vec_distance_threshold` | `hybrid_search.py` | 向量距离阈值（L2），融合前安全网 | - |
| `hybrid.rrf_score_threshold` | `hybrid_search.py` | RRF 融合分数阈值，融合后质量门 | - |
| `hybrid.index_roles` | `hybrid_search.py` | 索引的消息角色，默认排除 tool 输出 | - |
| `hybrid.min_content_length` | `hybrid_search.py` | 最低索引内容长度，null=不限 | - |
| `hybrid.batch_token_limit` | `hybrid_search.py` | 批量索引 token 预算，每组 API 调用不超过此值 | - |
| `hybrid.auto_index_threshold` | `hybrid_search.py` | 未索引消息数达到此阈值时自动触发后台索引 | - |

## 文件清单

```
hermes-agent-customized/
├── hermes_cli/config.py                        # DEFAULT_CONFIG: auxiliary.session_search
├── tools/
│   ├── session_search_tool.py                  # 路由 + 3个工具: session_search, rebuild_hybrid_index, hybrid_index_status
│   └── hybrid_search.py                        # 核心: BM25+Vector+RRF+Reranker+Rebuild+Diagnostics
├── plugins/
│   └── hybrid-search-indexer/                  # 插件: on_session_finalize 钩子
│       ├── plugin.yaml
│       └── __init__.py
└── scripts/
    └── index_embeddings.py                     # 手动批量索引
```

## 数据库 Schema

```sql
-- 自动创建（首次索引时）
CREATE VIRTUAL TABLE IF NOT EXISTS message_vec USING vec0(
    message_id INTEGER PRIMARY KEY,
    embedding FLOAT[1024],
    session_id TEXT
);
```

## 注册工具

| 工具名 | 功能 | 参数 | check_fn |
|--------|------|------|----------|
| `session_search` | 搜索历史会话 | query, role_filter, limit | 始终可用 |
| `rebuild_hybrid_index` | 重建向量索引（DROP+重建+全量索引） | 无 | check_hybrid_search_requirements |
| `hybrid_index_status` | 查看索引状态和诊断信息 | 无 | 始终可用 |

### session_search 输出格式

```json
{
  "success": true,
  "query": "部署问题",
  "engine": "hybrid",
  "results": [...],
  "count": 3,
  "diagnostics": {
    "bm25_hits": 5,
    "vector_hits": 3,
    "vec_available": true,
    "has_api_key": true
  }
}
```

- `engine`: `"bm25"` 或 `"hybrid"`，明确告知使用了哪个引擎
- `diagnostics`: 仅 hybrid 模式返回，包含各路搜索命中数和可用性信息
- `diagnostics.vector_hits == 0` 表示向量搜索未贡献结果

## 启用步骤

1. 安装依赖：`pip install sqlite-vec pysqlite3-binary`
2. 配置 config.yaml：
   ```yaml
   auxiliary:
     session_search:
       engine: "hybrid"   # 或 "auto"
   ```
3. 启用插件：`hermes plugins enable hybrid-search-indexer`
4. 设置 API Key：`embed_api_key` 或 `SILICONFLOW_API_KEY` 环境变量
5. 首次全量索引：`python scripts/index_embeddings.py index`

## 测试结果

### Hybrid vs BM25 对比（43 查询，3 语言场景）

| 引擎 | 命中率 | 平均耗时 |
|------|--------|---------|
| BM25 | 5% | ~5ms |
| Hybrid (BM25+Vec+RRF) | 100% | ~250ms |
| Hybrid+Reranker | 98% | ~730ms |

### Reranker 评估结论

- 43 查询中：1 改善、3 恶化、39 不变 → 净改善 -2
- 额外延迟 +479ms
- **结论：默认关闭（`use_reranker: false`）**

### vec_distance_threshold 分析

BAAI/bge-m3 (L2 distance) 的 distance 分布：

| 相关性 | Distance 范围 | 示例 |
|--------|-------------|------|
| 高度相关 | 0.72 ~ 0.85 | "苹果品种"→fruit: 0.72, "数据库调优"→db: 0.75 |
| 相关 | 0.84 ~ 1.0 | "how to deploy"→deploy: 0.84, "网站安全"→security: 0.85, "上线流程"→deploy: 0.97 |
| 不相关 | > 1.0 | deploy→security: 1.08, fruit→db: 1.07 |

**两层阈值设计**：
- `vec_distance_threshold: 1.2` — 融合前安全网，过滤 distance > 1.2 的明显不相关向量结果
- `rrf_score_threshold: 0.0` — 融合后质量门，默认不过滤；大数据量下可设为 0.01~0.016 过滤低质量结果

### rrf_score_threshold 分析

RRF score 依赖数据量，小数据集下区分度差：

| 数据量 | 正确结果 RRF | 错误结果 RRF | 区分度 |
|--------|------------|------------|--------|
| 小（8 session） | 0.0164 | 0.0159 | 极差（差 0.0005） |
| 大（1000+ session） | 0.0328（双路命中） | 0.005~0.010 | 好 |

因此 `rrf_score_threshold` 默认 0.0（不过滤），用户可按数据规模调整。

### E2E 测试（含 LLM 摘要）

8 查询完整测试（threshold=2.0，无过滤）：

| 查询 | Top1 | Distance | LLM 摘要 |
|------|------|----------|---------|
| 上线流程 | deploy ✅ | 0.974 | ✅ 中文摘要 |
| 数据库调优 | db ✅ | 0.751 | ✅ 中文摘要 |
| 网站安全 | security ✅ | 0.847 | ✅ 中文摘要 |
| 苹果电脑 | apple_tech ✅ | 0.918 | ✅ 中文摘要 |
| 苹果品种 | fruit ✅ | 0.723 | ✅ 中文摘要 |
| how to deploy | deploy ✅ | 0.841 | ✅ 英文摘要 |
| 红烧肉做法 | cooking ✅ | 0.753 | ✅ 中文摘要 |
| 日本旅游攻略 | travel ✅ | 0.810 | ✅ 中文摘要 |

Top1 准确率 100%，LLM 摘要全部成功。

## 实施进度

```
Phase 0: 原型实现 ✅
Phase 1: 修复设计缺陷 ✅
  ├─ 配置路径 → auxiliary.session_search
  ├─ engine 字段 → provider 回归 LLM
  ├─ embed_* 独立配置
  ├─ config 传递
  └─ QMD_* 环境变量清理

Phase 2: 自动索引 ✅
  ├─ 懒索引 _lazy_index_unindexed()
  ├─ 插件 hybrid-search-indexer
  └─ index_status()

Phase 3: 测试与稳定 ✅
  ├─ sqlite-vec 安全加载（3层 fallback） ✅
  ├─ BM25 vs Hybrid 对比测试 ✅
  ├─ Reranker 评估（不推荐默认开启） ✅
  ├─ vec_top_k + vec_distance_threshold ✅
  ├─ rrf_score_threshold 两层阈值设计 ✅
  ├─ index_roles 过滤（默认排除 tool 消息） ✅
  ├─ rebuild_hybrid_index 工具 ✅
  ├─ hybrid_index_status 工具 ✅
  ├─ search 输出 diagnostics 诊断信息 ✅
  ├─ BM25 输出也标注 engine ✅
  └─ E2E 测试（含 LLM 摘要） ✅

Phase 4: Bug 修复与日志 ✅
  ├─ Bug 1: check_hybrid_search_requirements 模块级引用 → wrapper 函数 ✅
  ├─ Bug 2: pysqlite3 row_factory → pysqlite.Row ✅
  ├─ Bug 3-5: tuple 当 dict 访问 → _row_get() helper ✅
  ├─ Bug 6: _bm25_search 依赖 row_factory → Bug 2 间接修复 ✅
  ├─ Bug 7: hybrid 空结果不 fallback BM25 → auto 模式检查 count > 0 ✅
  ├─ Bug 8: _db_path vs db_path → getattr 双重查找 ✅
  ├─ rebuild_hybrid_index/hybrid_index_status db=None → 加入 _AGENT_LOOP_TOOLS ✅
  ├─ session_search 调用日志记录 → _log_and_return() 包装所有返回路径 ✅
  └─ 增强日志：bm25/vec命中数、threshold过滤前后、LLM摘要片段 ✅

Phase 5: 优化 ⏭️
  ├─ 智能消息过滤
  ├─ 查询向量缓存
  └─ 孤立向量清理
```
