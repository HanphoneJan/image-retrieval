# 01 - 系统架构总览

> **阅读目标**: 理解系统整体架构、技术选型和核心设计决策

---

## 1. 整体架构

### 1.1 三层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                      应用层 (Flask)                                    │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Web UI  │  │  /search     │  │  /api/tools  │  │  /api/index  │ │
│  │  Jinja2  │  │  /api/search │  │  /api/tools/ │  │  /add/remove │ │
│  │          │  │  /api/chat   │  │   call       │  │  /merge      │ │
│  └────┬─────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
└───────┼────────────────┼─────────────────┼─────────────────┼─────────┘
        │                │                 │                 │
        ▼                ▼                 ▼                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    LLM 辅助层 (可选，优雅降级)                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  LLMInterface    │  │  ImageRetrieval  │  │  ToolUsingAgent  │   │
│  │  · expand_query  │  │  Tools           │  │  (演示用)        │   │
│  │  · explain_result│  │  · 4个工具       │  │                  │   │
│  │  · chat          │  │  · FC格式兼容    │  │                  │   │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────────────┘   │
│           │                     │                                     │
└───────────┼─────────────────────┼─────────────────────────────────────┘
            │                     │
            ▼                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       检索引擎层 (核心)                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  CLIPModel       │  │  IndexModule     │  │  IncrementalIdx  │   │
│  │  (ViT-B/32)      │  │  (Faiss IVF+PQ)  │  │  Manager         │   │
│  │                  │  │                  │  │  · buffer        │   │
│  │  图像→512维向量  │  │  训练/检索/add   │  │  · deleted_ids   │   │
│  │  文本→512维向量  │  │  CPU/GPU自动切换 │  │  · dual-search   │   │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘   │
│           │                     │                     │              │
│           └─────────────────────┴─────────────────────┘              │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.2 数据流

```
【用户输入】
    │
    ├── 文本: "a dog playing"
    │     │
    │     ▼
    │  CLIP.encode_text_by_string() ──→ 512维向量
    │
    ├── 图片路径: "cat.jpg"
    │     │
    │     ▼
    │  CLIP.encode_image_by_path() ──→ 512维向量 (L2归一化)
    │
    └── 图片数组: np.ndarray
          │
          ▼
       CLIP.encode_image_by_ndarray() ──→ 512维向量 (L2归一化)

              │
              ▼
       Faiss Index Search
       IVF4096: 找最近的 nprobe 个聚类中心
       PQ32x8: 用码本快速计算近似距离
              │
              ▼
       TopK (IDs + Distances)
              │
              ▼
       map_dict[ID] → 图片路径
              │
              ▼
       (可选) LLM 查询扩展/结果解释
              │
              ▼
       【返回结果】
```

---

## 2. 技术栈

| 层级 | 组件 | 技术选型 | 说明 |
|:---|:---|:---|:---|
| **应用层** | Web 框架 | Flask | 轻量级, 14 个 REST 端点 |
| | Agent 协议 | OpenAI Function Calling | 行业标准工具格式 |
| **LLM 辅助层** | LLM 接口 | OpenAI SDK | 兼容任意 OpenAI 格式 API |
| | 查询扩展 | Prompt 工程 | 多视角语义变体生成 |
| **检索引擎层** | 特征编码 | CLIP (ViT-B/32) | 图像+文本→512 维统一空间 |
| | 向量检索 | Faiss IVF4096+PQ32x8 | 倒排+量化, 64x 压缩 |
| | 增量索引 | IncrementalIndexManager | 双缓冲+逻辑删除 |
| | 深度学习 | PyTorch | CLIP 模型推理 |
| **基础设施** | 配置管理 | dataclass + dotenv | 环境变量覆盖 |
| | 数据持久化 | Pickle + JSON | 特征矩阵 + 索引状态 |
| | 图像处理 | OpenCV + Pillow | BGR 读取 → RGB 转换 |

---

## 3. 模块职责

| 模块 | 文件 | 职责 | 设计要点 |
|:---|:---|:---|:---|
| **CLIPModel** | [retrieval_by_faiss.py](../../retrieval_by_faiss.py) | 图像/文本→512 维向量 | 图像 L2 归一化, 文本不需要 |
| **IndexModule** | [retrieval_by_faiss.py](../../retrieval_by_faiss.py) | Faiss 索引管理 | index_factory 工厂创建, GPU/CPU 自动切换 |
| **ImageRetrievalModule** | [retrieval_by_faiss.py](../../retrieval_by_faiss.py) | 检索流程整合 | 统一文本/图像/ndarray 三种输入 |
| **IncrementalIndexManager** | [incremental_index_manager.py](../../incremental_index_manager.py) | 增量索引管理 | 双缓冲+逻辑删除, 状态持久化 |
| **ImageIndexBuilder** | [image_index_builder.py](../../image_index_builder.py) | 新图片特征提取 | 批量处理, 错误隔离 |
| **LLMInterface** | [llm_interface.py](../../llm_interface.py) | LLM 能力封装 | 优雅降级: `available` 属性 |
| **ImageRetrievalTools** | [agent_tools.py](../../agent_tools.py) | Agent 工具定义 | 4 个工具, OpenAI FC 格式兼容 |
| **Flask App** | [flask_app.py](../../flask_app.py) | HTTP 入口 | 14 个端点, 组件组装 |

---

## 4. 关键设计决策

### 4.1 为什么选 CLIP 而不是 ResNet？

| 维度 | ResNet | CLIP |
|:---|:---|:---|
| 特征空间 | 图像专用 | 图像+文本统一空间 |
| 检索能力 | 只能以图搜图 | 以文搜图 + 以图搜图 |
| 语义理解 | 需领域标注训练 | 零样本, 4 亿图文对预训练 |
| 特征维度 | 2048 维 | 512 维 (更紧凑) |

### 4.2 为什么选 IVF4096+PQ32x8？

来自 [demos/](../../demos/) 的基准测试结论：

| 索引类型 | 内存 (12万×512维) | 搜索速度 | 精度 |
|:---|:---|:---|:---|
| Flat (暴力) | 240 MB | 慢 (O(N)) | 100% |
| IVF4096 (纯倒排) | 240 MB | 快 (O(N/4096×nprobe)) | 可调 |
| IVF4096+PQ32x8 | ~4 MB | 快 + 距离计算加速 | 可调 |

- IVF4096: `4×√120000 ≈ 1386`, 取 4096 是 Faiss 推荐的 2 的幂
- PQ32x8: 32 段 × 8bit = 32 bytes/向量, 原始 2048 bytes → **64 倍压缩**
- 权衡: nprobe 增大 → recall↑ + 耗时↑, 这是可以运行时调节的

### 4.3 为什么需要 L2 归一化？

```
CLIP 输出未归一化, 但我们关心余弦相似度:

cos_sim(A,B) = (A·B) / (||A|| × ||B||)

L2 归一化后 ||A||=||B||=1:
cos_sim(A,B) = A·B                    (余弦相似度 = 内积)

Faiss 用 L2 距离:
||A-B||² = ||A||² + ||B||² - 2(A·B) = 2 - 2cos_sim

结论: L2 距离越小 ↔ 余弦相似度越大
      → Faiss L2 检索结果即语义最相似结果
```

代码: [retrieval_by_faiss.py:138](../../retrieval_by_faiss.py#L138) — 图像必须归一化, [retrieval_by_faiss.py:168](../../retrieval_by_faiss.py#L168) — 文本不需要 (CLIP 内部已处理)。

### 4.4 优雅降级

```python
# llm_interface.py:43-44
self._available = bool(self.api_key)

# 调用方
self.enable_expansion = enable_expansion and self.llm.available
```

LLM_API_KEY 为空 → `available=False` → 所有 LLM 功能自动跳过, 退化为纯 CLIP+Faiss 检索。

---

## 5. API 端点一览

| 端点 | 方法 | 功能 | LLM 依赖 |
|:---|:---|:---|:---|
| `/` | GET | Web UI | 无 |
| `/search` | POST | 传统图像检索 | 无 |
| `/api/search/rag` | POST | LLM 辅助检索 | 可选 |
| `/api/search/expand` | POST | 查询扩展 | 需要 |
| `/api/rag/qa` | POST | 基于检索结果的问答 | 需要 |
| `/api/chat` | POST | 多轮对话 | 需要 |
| `/api/tools` | GET | 获取工具列表 | 需要 |
| `/api/tools/call` | POST | 调用 Agent 工具 | 需要 |
| `/api/index/status` | GET | 索引状态 | 无 |
| `/api/index/add` | POST | 增量添加图片 | 无 |
| `/api/index/remove` | POST | 逻辑删除图片 | 无 |
| `/api/index/merge` | POST | buffer→主索引合并 | 无 |
| `/api/index/rebuild` | POST | 全量重建索引 | 无 |
| `/api/status` | GET | 系统综合状态 | 无 |

---

## 6. 增量索引架构

### 6.1 问题

Faiss IVF 索引训练好后:
- 添加单张图片 → 传统做法需重建全量索引 (12 万图需 2-4 小时)
- 删除图片 → Faiss 根本没有 delete 接口

### 6.2 方案

```
┌──────────────────────────────────────────────┐
│           IncrementalIndexManager             │
│                                               │
│  ┌─────────────────┐  ┌─────────────────┐    │
│  │  主索引 (Faiss)  │  │  缓冲区 (Python) │    │
│  │  IVF4096+PQ32x8 │  │  List[向量]      │    │
│  │  存量数据        │  │  新增数据        │    │
│  │  Faiss 快速检索  │  │  np.dot 暴力搜索 │    │
│  └────────┬────────┘  └────────┬────────┘    │
│           │                    │              │
│           └────────┬───────────┘              │
│                    ▼                          │
│           合并排序 → 过滤 deleted_ids → topk  │
│                                               │
│  ┌─────────────────────────────────────┐     │
│  │  deleted_ids: Set[int]              │     │
│  │  逻辑删除标记, 搜索时 O(1) 过滤     │     │
│  └─────────────────────────────────────┘     │
└──────────────────────────────────────────────┘
```

| 操作 | 策略 | 复杂度 |
|:---|:---|:---|
| Add | 进 buffer, 不动主索引 | O(1) |
| Remove | ID 加入 deleted_ids 集合 | O(1) |
| Search | 主索引 + buffer 双路 → 合并排序 → 过滤 → topk | O(Faiss + B×d) |
| Merge | buffer vstack → Faiss add (免训) | O(B×CLIP) |
| Rebuild | 过滤 deleted → 新建 IndexModule + train | O(N×CLIP) |

---

## 7. 架构演进方向

```
当前
 │
 ├── [x] Faiss IVF+PQ 索引 + CLIP 跨模态编码
 ├── [x] 双缓冲增量索引 (add/remove/search/merge/rebuild)
 ├── [x] Function Calling 格式 Agent 工具
 ├── [x] LLM 查询扩展 + 结果解释
 │
 ├── [ ] 重排序模型 (Cross-Encoder 精排)
 ├── [ ] 混合检索 (BM25 关键词 + CLIP 向量)
 ├── [ ] 多租户索引隔离
 └── [ ] 分布式向量数据库 (Milvus)
```

---

**下一章**: [02 - 核心系统设计](./02-core-system-design.md) — 三大模块深入拆解
