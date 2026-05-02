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
  │      └─ _start_auto_index_daemon() 启动后台守护线程
  │   3. search(query)
  │      ├─ BM25: db.search_messages() (FTS5)
  │      ├─ Vector: sqlite-vec KNN (vec_top_k + vec_distance_threshold)
  │      ├─ RRF: 融合排序
  │      ├─ RRF score threshold 过滤（可选，默认不过滤）
  │      └─ Reranker: 可选重排序
  │   4. LLM 摘要（auxiliary_client，配置独立）
  │
  └─ engine == "auto" → 尝试 hybrid，失败回退 bm25
```

## 索引机制

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: 后台守护线程自动索引（Auto Index Daemon）                │
│  触发：HybridSessionSearch 初始化时启动，独立于搜索运行            │
│  逻辑：每 auto_index_interval 秒检查未索引消息数                   │
│        未索引 >= auto_index_threshold → _fetch_unindexed          │
│        → index_batch（按 token 分组，批量 API + 批量 DB 写入）     │
│  特性：_indexing_lock 防并发，_stop_event 支持优雅停止             │
│  优点：不依赖搜索触发，不遗漏，不阻塞主线程                        │
│  配置：auto_index=true, auto_index_threshold=10,                  │
│        auto_index_interval=30                                     │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: 会话结束索引（Session Finalize Indexing）               │
│  触发：on_session_finalize 钩子（CLI退出、/new、/reset等）         │
│  实现：plugins/hybrid-search-indexer/ 插件                        │
│  逻辑：index_session(session_id) → _index_unindexed_batch         │
│  特性：限定 session_id，只索引该 session 的未索引消息              │
│  优点：session 结束后立即可供搜索                                  │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: 手动触发索引（Manual Indexing）                         │
│  触发：index_unindexed_messages 工具 / rebuild_hybrid_index 工具  │
│  区别：index_unindexed_messages 增量索引，rebuild 重建全量索引     │
│  参数：session_id 可选，限定某个 session                           │
└─────────────────────────────────────────────────────────────────┘

所有路径统一经过：
  _fetch_unindexed(session_id?, limit?) → 查询未索引消息
  → EmbeddingIndexer.index_batch(items) → 按 token 分组
  → _call_embedding_api_batch(texts) → 批量 embedding API
  → 批量 INSERT INTO message_vec
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
      auto_index: true
      auto_index_threshold: 10
      auto_index_interval: 30
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
| `hybrid.auto_index` | `hybrid_search.py` | 自动索引总开关，false=禁用守护线程 | - |
| `hybrid.auto_index_threshold` | `hybrid_search.py` | 未索引消息数达到此阈值时触发索引 | - |
| `hybrid.auto_index_interval` | `hybrid_search.py` | 守护线程检查间隔（秒），推荐 10~300 | - |

### 配置填写规范

| 字段 | 类型 | 默认值 | 范围/规范 | 说明 |
|------|------|--------|-----------|------|
| `engine` | string | `"bm25"` | `"bm25"` / `"hybrid"` / `"auto"` | auto=尝试hybrid，失败回退bm25 |
| `vec_distance_threshold` | float | `1.2` | `0.5 ~ 2.0` | bge-m3 L2距离：相关<1.0，弱相关1.0~1.2，不相关>1.2 |
| `rrf_score_threshold` | float | `0.0` | `0.0 ~ 0.05` | 小数据集区分度差，建议0.0；大数据集可设0.01~0.016 |
| `index_roles` | list | `["user","assistant"]` | 任意 role 组合 | 排除 tool 输出可节省 API 成本和减少噪声 |
| `min_content_length` | int/null | `null` | `null` 或正整数 | null=不限，任何长度都索引；设为50则跳过短消息 |
| `batch_token_limit` | int | `7000` | `1000 ~ 8000` | bge-m3 上限 8192 tokens，7000留安全余量 |
| `auto_index` | bool | `true` | `true` / `false` | false=禁用守护线程，只能手动触发索引 |
| `auto_index_threshold` | int | `10` | `1 ~ 1000` | 未索引消息达到此数量时触发批量索引 |
| `auto_index_interval` | int | `30` | `5 ~ 3600`（秒） | 守护线程轮询间隔；<5无意义（COUNT查询有开销）；>300可能延迟过大 |

## 文件清单

```
hermes-agent-customized/
├── hermes_cli/config.py                        # DEFAULT_CONFIG: auxiliary.session_search
├── tools/
│   ├── session_search_tool.py                  # 路由 + 4个工具
│   └── hybrid_search.py                        # 核心: BM25+Vector+RRF+Reranker+Daemon+Diagnostics
├── plugins/
│   └── hybrid-search-indexer/                  # 插件: on_session_finalize 钩子
│       ├── plugin.yaml
│       └── __init__.py
├── model_tools.py                              # _AGENT_LOOP_TOOLS 注册
├── run_agent.py                                # agent 循环直接调用（注入 db）
├── toolsets.py                                 # 工具集声明
└── scripts/
    └── index_embeddings.py                     # 手动批量索引脚本
```

## 数据库 Schema

```sql
-- 自动创建（首次索引时）
CREATE VIRTUAL TABLE IF NOT EXISTS message_vec USING vec0(
    message_id INTEGER PRIMARY KEY,
    embedding FLOAT[1024],
    session_id TEXT
);

-- messages 表已有字段（由 hermes_state.py 管理）
-- token_count INTEGER  ← 批量索引时用于 token 估算
```

## 注册工具

| 工具名 | 功能 | 参数 | check_fn |
|--------|------|------|----------|
| `session_search` | 搜索历史会话 | query, role_filter, limit | 始终可用 |
| `rebuild_hybrid_index` | 重建向量索引（DROP+全量索引） | 无 | check_hybrid_search_requirements |
| `hybrid_index_status` | 查看索引状态和诊断信息 | 无 | 始终可用 |
| `index_unindexed_messages` | 增量索引未索引的消息 | session_id（可选） | check_hybrid_search_requirements |

所有 4 个工具均在 `_AGENT_LOOP_TOOLS` 中，由 `run_agent.py` 的 agent 循环直接调用，注入 `db=self._session_db`。

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
    "vec_before_threshold": 8,
    "vec_distance_threshold": 1.2,
    "fused_before_rrf_threshold": 6,
    "fused_after_rrf_threshold": 4,
    "rrf_score_threshold": 0.0,
    "vec_available": true,
    "has_api_key": true
  }
}
```

- `engine`: `"bm25"` 或 `"hybrid"`，明确告知使用了哪个引擎
- `diagnostics`: hybrid 模式返回，包含各路搜索命中数、阈值过滤前后对比、可用性信息
- `vec_before_threshold` / `vector_hits`: 向量搜索原始命中数 / 经过 vec_distance_threshold 过滤后数量
- `fused_before_rrf_threshold` / `fused_after_rrf_threshold`: RRF 融合后 / rrf_score_threshold 过滤后数量

### session_search 日志格式

每次调用自动记录到 agent.log：

```
# hybrid 成功
session_search query='deploy database' engine=hybrid count=2 bm25=5 vec_raw=8 vec_filtered=3 vec_thresh=1.2 fused_before=6 fused_after=4 summaries=['User discussed deploying...', 'Discussion about CI/CD...'] elapsed=0.25s

# BM25 成功
session_search query='deploy' engine=bm25 count=3 bm25=5 elapsed=0.35s

# 失败
session_search query='test' engine=bm25 count=0 error=Session database not available. elapsed=0.00s
```

## 启用步骤

1. 安装依赖：`pip install sqlite-vec pysqlite3`
2. 配置 config.yaml：
   ```yaml
   auxiliary:
     session_search:
       engine: "hybrid"   # 或 "auto"
   ```
3. 启用插件：`hermes plugins enable hybrid-search-indexer`
4. 设置 API Key：`embed_api_key` 或 `SILICONFLOW_API_KEY` 环境变量
5. 首次全量索引（可选，守护线程会自动处理）：`python scripts/index_embeddings.py index`

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

Phase 5: 批量索引与智能触发 ✅
  ├─ Embedding API 批量调用（input 数组） ✅
  ├─ 按 token 分组（batch_token_limit） ✅
  ├─ token_count 字段利用（DB 已有） ✅
  ├─ _truncate_to_token_budget 替代固定字符截断 ✅
  ├─ 批量 400 自动降级逐条调用 ✅
  ├─ min_content_length 可配置（默认 null=不限） ✅
  ├─ 后台守护线程（auto_index_daemon）替代懒加载 ✅
  ├─ auto_index_threshold 阈值触发 ✅
  ├─ auto_index_interval 轮询间隔 ✅
  ├─ _indexing_lock 防并发 ✅
  ├─ _fetch_unindexed 统一查询（替代3处重复SQL） ✅
  ├─ _index_unindexed_batch 统一批处理入口 ✅
  ├─ index_unindexed_messages 工具 ✅
  ├─ 移除 enqueue 队列（不再有丢弃消息问题） ✅
  └─ 移除 _index_one 逐条索引（全部走 batch） ✅

Phase 6: 优化 ⏭️
  ├─ 查询向量缓存
  └─ 孤立向量清理
```
