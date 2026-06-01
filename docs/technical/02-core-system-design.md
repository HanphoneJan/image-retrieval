# 02 - 核心系统设计

> **阅读目标**: 深入理解三大核心模块的设计思路和实现细节

---

## 三大模块对照简历

| 简历关键词 | 本章节 | 核心设计 |
|---|---|---|
| Faiss IVF4096+PQ32x8 | §1 检索引擎 | CLIP 编码 → Faiss 索引 → 统一检索接口 |
| 双缓冲增量索引 | §2 增量索引 | 主索引 + buffer + 逻辑删除 + 状态持久化 |
| Function Calling + LLM | §3 LLM 辅助 | 查询扩展 + 结果解释 + Agent 工具 |

---

## 1. 检索引擎

### 1.1 CLIPModel — 多模态编码器

**职责**: 图像和文本 → 512 维向量, 同一空间可直接比较

```python
class CLIPModel:
    def encode_image_by_path(self, path_img: str) -> np.ndarray    # 图像路径 → 1×512
    def encode_image_by_ndarray(self, image_rgb: np.ndarray) -> np.ndarray  # 数组 → 1×512
    def encode_text_by_string(self, text: str) -> np.ndarray       # 文本 → 1×512
```

**图像编码流程**:

```
图片路径 → cv_imread (BGR) → BGR→RGB → PIL.Image
  → CLIP preprocess (Resize 224 + Normalize)
  → model.encode_image() → L2 归一化 → (1, 512) float32
```

关键细节: 图像编码后必须 L2 归一化 (`feat /= feat.norm(dim=-1, keepdim=True)`), 文本不需要。原因: CLIP 训练时文本编码器内部已做归一化, 图像编码器没有。

### 1.2 IndexModule — Faiss 索引管理

**职责**: 封装 Faiss 索引的创建、训练、检索、增量添加

```python
class IndexModule:
    def __init__(self, index_string, feat_dim, feat_mat):
        # 1. faiss.index_factory(512, "IVF4096,PQ32x8")  → 创建索引
        # 2. GPU 可用 → index_cpu_to_gpu()
        # 3. index.train(feat_mat)   → KMeans 聚类 + PQ 码本训练
        # 4. index.add(feat_mat)     → 添加全量向量

    def feat_retrieval(self, feat_query, topk):
        # distance, ids = index.search(feat_query, topk)
        # 返回 L2 距离 + Faiss 内部 ID

    def add_vectors(self, vectors):
        # 追加向量到已有索引（免重新训练, Faiss IVADC 支持）
```

**IVF4096+PQ32x8 参数含义**:

```
IVF4096,PQ32x8
  │      │   │
  │      │   └── 8 bit 量化 = 256 个码本中心/段
  │      └────── 32 个子向量段 (512/32 = 16 维/段)
  └────────────── 4096 个倒排列表 (KMeans 聚类中心)

内存: 每向量 32 段 × 1 byte = 32 bytes
      原始 512 × 4 = 2048 bytes → 压缩 64 倍
```

### 1.3 ImageRetrievalModule — 检索入口

**职责**: 组装 CLIP + Faiss, 提供统一检索接口

```python
class ImageRetrievalModule:
    def retrieval_func(self, query_info, topk):
        # 1. 自动判断输入类型:
        #    os.path.exists(query_info)  → encode_image_by_path  (以图搜图)
        #    isinstance(query_info, str) → encode_text_by_string (以文搜图)
        #    isinstance(query_info, ndarray) → encode_image_by_ndarray
        #
        # 2. feat_retrieval(feat_vec, topk)
        #
        # 3. map_dict[faiss_id] → 图片路径
        return distance, ids, path_list
```

---

## 2. 增量索引

### 2.1 数据结构

```python
class IncrementalIndexManager:
    # 核心数据:
    self.main_index: IndexModule        # Faiss IVF 主索引 (存量数据)
    self.buffer_features: List[ndarray] # 缓冲区特征 (新增数据)
    self.buffer_paths: List[str]        # 缓冲区路径
    self.buffer_ids: List[int]          # 缓冲区 ID
    self.deleted_ids: Set[int]          # 逻辑删除标记
    self.next_id: int                   # 自增 ID 计数器
    self.id_to_path: Dict[int, str]     # ID→路径 映射
```

### 2.2 add_images — 添加图片

```
image_paths → ImageIndexBuilder.extract_features() → CLIP 编码
  → 分配 ID (next_id 自增)
  → 追加到 buffer_features / buffer_paths / buffer_ids
  → 检查: len(buffer) >= threshold? → 提示 merge
  → 返回 {"added": N, "ids": [id1, id2, ...]}
```

不碰主索引, O(1) 完成。复杂度主要在新图片的 CLIP 编码 (~100ms/张)。

### 2.3 remove_images — 逻辑删除

```
image_ids → 逐个检查:
  - 存在于主索引 OR 缓冲区? → deleted_ids.add(id)
  - 恰好在缓冲区? → 直接从 list pop 移除
  - 都不存在? → 返回 not_found
```

为什么是逻辑删除?

| 方案 | 可行性 |
|:---|:---|
| 物理删除 | Faiss IVF 索引不支持逐条 delete |
| 逻辑删除 | O(1) 标记, 搜索时 O(1) 过滤, 积累多了 rebuild 清理 |

### 2.4 search — 双路搜索合并

```python
def search(self, query_vector, topk):
    # 1. 主索引搜索
    main_dist, main_ids = main_index.feat_retrieval(query_vector, topk*2)

    # 2. 缓冲区暴力搜索
    buffer_matrix = np.vstack(buffer_features)  # (M, 512)
    similarities = np.dot(buffer_matrix, query_vector.T).flatten()
    buffer_dist = 1 - similarities
    # → argsort → topk*2

    # 3. 合并
    for dist, id in (主索引结果 + 缓冲区结果):
        if id not in deleted_ids:    # 过滤逻辑删除
            all_results.append((dist, id, path))

    # 4. 全局排序 → topk
    all_results.sort(key=lambda x: x[0])
    return all_results[:topk]
```

**缓冲区为什么用暴力 dot product 而不是也建 Faiss 索引?**

- buffer 通常 < 1000 条, 512 维: `np.dot` 只需几十微秒
- 超了就 merge 到主索引, 没必要为 buffer 单独建索引
- 如果一定要优化: 改用 `faiss.IndexFlatIP` (内积索引), BLAS 加速比 numpy 快 2-3 倍

### 2.5 merge_buffer — 缓冲区合并

```
buffer vstack → float32 矩阵 (M, 512)
  → main_index.add_vectors(buffer_matrix)  # Faiss IVADC 免训
  → 清空 buffer
  → 保存状态
```

**为什么 add_vectors 不需要重新训练?**

```
IVF 层: 4096 个聚类中心在初始训练时就固定了
       新向量只需找最近的聚类中心 → 放入对应倒排列表
       聚类中心本身不移动

PQ 层: 32×256 个码本中心在初始训练时就固定了
       新向量按已有码本编码即可
       码本不更新
```

类比: 行政区划 (聚类中心) 画好后不改, 新居民 (新向量) 直接落户最近行政区就行。代价是: 如果新数据分布变了, 编码质量会下降 → 所以需要定期 rebuild 刷新。

### 2.6 rebuild_index — 完全重建

```
all_features → 过滤 deleted_ids → 只保留有效向量
  → 新建 IndexModule(索引字符串, feat_dim, 有效特征)
     → index.train() → 重新 KMeans + PQ 训练
  → 清空 buffer + deleted_ids + 重置 next_id
  → 主索引替换为新索引
```

这是唯一真正物理删除数据的操作。训练成本与数据量成正比 (12 万图约 2 分钟)。

### 2.7 状态持久化

```python
def save_state(self):
    # JSON: next_id, deleted_ids, buffer_ids, buffer_paths
    # Pickle: buffer_features (vstack 后存)

def load_state(self, map_dict):
    # 恢复 JSON 状态 + Pickle 特征
    # buffer_features 重建为 List[ndarray] 格式
```

每次 add/remove/merge 后自动保存, Flask 启动时自动加载。JSON 可读便于调试, Pickle 高效存 ndarray。

---

## 3. LLM 辅助层

### 3.1 LLMInterface — LLM 能力封装

**职责**: 统一管理 LLM 调用, 提供查询扩展和结果解释

```python
class LLMInterface:
    @property
    def available(self) -> bool:       # 关键: 优雅降级开关
        return bool(self.api_key)

    def expand_query(self, user_query, num_expansions=3) -> List[str]:
        # Prompt: 生成 N 个不同表达的搜索意图
        # 失败 → 返回 [user_query] (降级)

    def explain_results(self, query, results) -> str:
        # Prompt: 解释检索结果与查询的相关性
        # 失败 → 返回空 (降级)
```

**查询扩展 Prompt 设计**:
```
作为图像搜索助手, 请基于用户的查询生成 3 个不同表达的搜索意图

用户查询: "a dog playing"

要求:
1. 保持原始语义, 但使用不同的词汇和表达方式
2. 可以从不同角度描述 (风格、场景、对象、颜色等)
3. 每个扩展查询简洁明了, 不超过 20 个字

输出 (每行一个):
a dog playing in the park
puppy running on grass
canine enjoying outdoors
```

### 3.2 ImageRetrievalTools — Agent 工具集

**职责**: 将检索能力封装为标准 Agent 工具, 兼容 OpenAI Function Calling 格式

4 个内置工具:

| 工具名 | 功能 | 关键参数 |
|:---|:---|:---|
| search_by_text | 文本搜图 | query, topk |
| search_by_image | 以图搜图 | image_path, topk |
| explain_search_results | 解释检索结果 | original_query, results |
| answer_with_image_context | 基于检索图片问答 | question, context_size |

**Function Calling JSON Schema** (以 search_by_text 为例):

```json
{
  "type": "function",
  "function": {
    "name": "search_by_text",
    "description": "根据文本描述搜索相关图片",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "搜索文本描述"},
        "topk": {"type": "integer", "description": "返回结果数量, 默认5"}
      },
      "required": ["query"]
    }
  }
}
```

**工具执行流程**:
```
POST /api/tools/call  {"tool": "search_by_text", "arguments": {...}}
  → tools.execute_tool(tool_name, arguments)
    → 查找 tool._tools[name] → func(**arguments)
    → 返回 {"tool": ..., "result": ..., "success": true/false}
```

### 3.3 ToolUsingAgent — 工具使用演示

**职责**: 演示 LLM 如何自主选择工具。构建 system prompt 包含工具描述 → LLM 返回 JSON 格式工具调用 → 解析执行 → 返回结果。

```python
class ToolUsingAgent:
    def run(self, user_input):
        # 1. system_prompt = "你是图像检索助手。可用工具: ... 返回JSON格式工具调用"
        # 2. LLM 推理 → 解析 JSON → 提取 tool_name + arguments
        # 3. tools.execute_tool(tool_name, arguments)
        # 4. 返回 {user_input, tool_call, result}
```

### 3.4 查询扩展 + 结果解释完整流程

```
用户查询 "a dog playing"
  │
  ▼
LLM.expand_query() → ["a dog playing", "puppy running", "canine outdoor"]
  │
  ▼
对每个查询 → CLIP 编码 → Faiss 检索 (topk*2)
  │
  ▼
去重融合: Set(path) 去重 → 按 distance 排序 → 截断 topk
  │
  ▼
LLM.explain_results(query, top_results) → 自然语言解释
  │
  ▼
返回 {original_query, expanded_queries[], results[], ai_explanation}
```

**去重策略**: 同一图片可能被多个扩展查询召回, 用 Set 记录已出现 path, 重复的跳过 (保留首次出现的距离值)。

---

## 4. 配置管理

```python
# config/base_config.py
@dataclass
class Config:
    # 模型
    clip_backbone_type: str = "ViT-B/32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 索引
    index_string: str = "IVF4096,PQ32x8"
    feat_dim: int = 512  # ViT-B/32 → 512, ViT-L/14 → 768

    # LLM (环境变量)
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

    # 功能开关
    rag_enable_expansion: bool = True
    rag_enable_explanation: bool = True
```

优先级: 环境变量 > .env 文件 > 代码默认值。

---

**上一章**: [01 - 架构总览](./01-architecture-overview.md)
**下一章**: [03 - 关键算法实现](./03-key-algorithms.md) — IVF/PQ/双缓冲算法详解
