# Hybrid Session Search — Agent 使用指南

## 可用工具

### 1. `session_search` — 搜索历史会话

搜索过去的对话记录，支持 BM25 关键词搜索和 Hybrid 混合搜索。

**参数：**
- `query`（必填）：搜索关键词
- `role_filter`（可选）：只搜索特定角色，如 `"user,assistant"` 跳过 tool 输出
- `limit`（可选）：返回条数，默认 3

**输出格式：**
```json
{
  "success": true,
  "query": "部署问题",
  "engine": "hybrid",
  "results": [
    {
      "session_id": "abc123",
      "when": "2 hours ago",
      "source": "cli",
      "model": "gpt-4o",
      "hybrid_score": 0.0328,
      "summary": "用户讨论了微服务部署的CI/CD流程..."
    }
  ],
  "count": 1,
  "diagnostics": {
    "bm25_hits": 5,
    "vector_hits": 3,
    "vec_available": true,
    "has_api_key": true
  }
}
```

**关键字段解读：**
- `engine`: `"bm25"` 或 `"hybrid"` — 当前使用的搜索引擎
- `diagnostics.vector_hits`: 向量搜索命中数。如果为 0，说明向量搜索未贡献结果（可能索引为空或 API 不可用）
- `diagnostics.vec_available`: sqlite-vec 是否可用
- `diagnostics.has_api_key`: embedding API 密钥是否配置
- `hybrid_score`: RRF 融合分数，越高表示两路搜索都认为该结果相关

### 2. `hybrid_index_status` — 查看索引状态

检查 hybrid 搜索的索引状态，用于诊断搜索问题。

**参数：** 无

**输出格式：**
```json
{
  "available": true,
  "total_indexable": 100,
  "indexed": 50,
  "unindexed": 50,
  "indexing_progress": "50/100",
  "queue_size": 0,
  "indexed_count": 50,
  "failed_count": 0,
  "vec_available": true,
  "index_roles": ["user", "assistant"],
  "engine_config": "hybrid"
}
```

**使用场景：**
- 搜索结果为空时，检查 `indexed` 是否为 0（索引未建立）
- 搜索质量差时，检查 `unindexed` 是否很多（需要重建索引）
- 排查问题时，检查 `available`、`vec_available`、`engine_config`

### 3. `rebuild_hybrid_index` — 重建向量索引

从零重建向量索引（删除旧索引 → 重新创建 → 全量索引）。

**参数：** 无

**输出格式：**
```json
{
  "rebuilt": true,
  "total_messages": 100,
  "indexed": 98,
  "failed": 2,
  "index_roles": ["user", "assistant"]
}
```

**使用场景：**
- 修改 `index_roles` 配置后，需要重建索引
- 索引损坏或数据不一致时
- 大量新消息未被索引时

**注意：** 大型数据库可能需要几分钟完成。

---

## 搜索引擎说明

| engine 值 | 行为 |
|-----------|------|
| `bm25` | 纯关键词搜索（FTS5），快速但中文同义词/语义搜索差 |
| `hybrid` | BM25 + 向量语义搜索 + RRF 融合，召回率高 |
| `auto` | 尝试 hybrid，sqlite-vec 不可用时回退 bm25 |

引擎由 `config.yaml` 中 `auxiliary.session_search.engine` 决定，agent 无法在调用时切换。

---

## 诊断流程

当 `session_search` 返回结果不理想时：

1. **检查 engine** — 输出中 `engine` 是 `"bm25"` 还是 `"hybrid"`？
   - 如果是 `"bm25"`，说明 hybrid 未启用或不可用
   
2. **检查 diagnostics** — hybrid 模式下：
   - `vector_hits == 0` → 向量搜索未贡献结果，检查索引状态
   - `vec_available == false` → sqlite-vec 未安装或加载失败
   - `has_api_key == false` → 未配置 embedding API 密钥

3. **调用 `hybrid_index_status`** — 查看详细索引状态
   - `indexed == 0` → 索引为空，需要等待懒索引或手动重建
   - `unindexed` 很多 → 调用 `rebuild_hybrid_index` 重建

4. **重建索引** — 调用 `rebuild_hybrid_index`

---

## 配置参考

所有配置在 `~/.hermes/config.yaml` 的 `auxiliary.session_search` 下：

```yaml
auxiliary:
  session_search:
    engine: "hybrid"              # 搜索引擎: bm25 | hybrid | auto
    hybrid:
      embed_model: "BAAI/bge-m3"  # Embedding 模型
      embed_base_url: "https://api.siliconflow.cn/v1"
      embed_api_key: ""            # 或用 SILICONFLOW_API_KEY 环境变量
      vec_distance_threshold: 1.2  # 向量距离阈值（L2），融合前安全网
      rrf_score_threshold: 0.0     # RRF 分数阈值，融合后质量门
      index_roles: ["user", "assistant"]  # 索引哪些角色的消息
      use_reranker: false          # 是否启用 reranker
```
