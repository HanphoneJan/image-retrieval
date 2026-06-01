# 多模态图像检索引擎 - 技术文档导航

> **项目定位**: 基于 CLIP + Faiss + LLM 的图像检索引擎
> **简历主线**: ① Faiss IVF+PQ 跨模态检索 ② 双缓冲增量索引 ③ Function Calling 集成 LLM
> **代码规模**: ~2,800 行 Python，8 个核心模块

---

## 简历描述（最终定稿）

> 基于 CLIP 多模态编码器，采用 Faiss IVF4096+PQ32x8 索引实现跨模态语义检索；设计双缓冲增量索引架构解决向量索引动态增删问题；通过 Function Calling 集成 LLM 实现查询扩展与结果解释。

三条主线对应关系：

| 简历关键词 | 对应模块 | 核心文件 |
|---|---|---|
| Faiss IVF4096+PQ32x8 | 向量检索引擎 | [retrieval_by_faiss.py](../../retrieval_by_faiss.py) |
| 双缓冲增量索引 | 增量索引管理 | [incremental_index_manager.py](../../incremental_index_manager.py) |
| Function Calling + LLM | LLM 接口 + Agent 工具 | [llm_interface.py](../../llm_interface.py), [agent_tools.py](../../agent_tools.py) |

---

## 文档清单

| 序号 | 文档 | 内容概要 | 阅读时间 |
|:---:|:---|:---|:---:|
| 00 | **本导航** | 文档导览、核心亮点、面试话术速记 | 10分钟 |
| 01 | [架构总览](./01-architecture-overview.md) | 系统分层架构、技术选型、设计决策 | 20分钟 |
| 02 | [核心系统设计](./02-core-system-design.md) | 三大核心模块详解、接口设计 | 40分钟 |
| 03 | [关键算法实现](./03-key-algorithms.md) | IVF/PQ 原理、双缓冲算法、复杂度分析 | 30分钟 |
| 04 | [小白概念指南](./04-core-concepts-for-beginners.md) | 大白话解释、类比理解 | 25分钟 |
| 05 | [面试准备指南](./05-interview-preparation.md) | 话术、高频问题、自查清单 | 60分钟 |

---

## 核心亮点速览

### 1. CLIP + Faiss 跨模态语义检索

- CLIP (ViT-B/32) 将图像和文本编码到同一 512 维向量空间
- Faiss IVF4096+PQ32x8 索引：IVF 倒排缩小搜索范围 + PQ 乘积量化实现 64 倍向量压缩
- 支持以文搜图、以图搜图，三种输入（文本/图片路径/ndarray）统一接口

### 2. 双缓冲增量索引架构

```
主索引 (Faiss IVF, 已训练)  +  缓冲区 (Python list, 暴力搜索)  +  删除集合 (Set, 逻辑删除)
```

- **Add**: 新图片特征直接进 buffer，O(1) 完成，不动主索引
- **Remove**: 逻辑删除（ID 加入 deleted_ids 集合），搜索时过滤
- **Search**: 主索引 + buffer 双路搜索，合并排序后取 topk
- **Merge**: buffer 超阈值时追加到主索引（Faiss IVADC 支持不重新训练的 add）
- **Rebuild**: 删除积累足够多时，过滤 deleted_ids，全量重建新索引

### 3. Function Calling 集成 LLM

- 4 个 Agent 工具：search_by_text、search_by_image、explain_search_results、answer_with_image_context
- 完整兼容 OpenAI Function Calling JSON Schema 格式
- LLM 查询扩展：用户输入 → LLM 生成 3 个语义变体 → 多路检索 → 去重融合
- LLM 结果解释：检索结果 → LLM 生成自然语言说明
- 优雅降级：LLM_API_KEY 未配置时，自动退化为传统检索

---

## 面试话术速记

### 1 分钟版本

```
这是一个基于 CLIP + Faiss + LLM 的图像检索引擎。

三条主线：
第一，用 CLIP 把图像和文本编码到同一向量空间，Faiss IVF4096+PQ32x8 索引实现
    64 倍压缩的跨模态语义检索；
第二，设计双缓冲增量索引架构——主索引 + 内存 buffer + 逻辑删除集，
    解决 Faiss 索引不支持动态增删的问题；
第三，通过 Function Calling 集成 LLM，实现查询扩展和结果解释。

代码约 2800 行，8 个核心模块，每个模块可独立测试。
```

### 3 分钟版本

```
这个项目是一个多模态图像检索引擎，核心是"用自然语言搜图，用图搜相似图"。

【检索引擎】
CLIP (ViT-B/32) 把图像和文本都编码成 512 维向量，在同一个向量空间里比较相似度。
Faiss 索引用 IVF4096+PQ32x8——IVF 倒排索引把搜索范围缩小到最近的几个聚类，
PQ 乘积量化把每个 512 维向量从 2048 字节压缩到 32 字节，压缩比 64 倍。

【增量索引】
这是我花最多心思设计的部分。Faiss IVF 索引训练好后不支持原地增删——
加一张图重建全量索引太慢，删一张图根本没接口。
我的方案是三件套：主索引（Faiss IVF）负责存量数据的高效检索，
内存 buffer（Python list）接收新增数据做暴力搜索，
deleted_ids 集合做逻辑删除标记。
增删都是 O(1)，搜索时双路查询合并排序，buffer 超阈值就 merge 到主索引，
删除积累多了就 rebuild 物理清理。

【LLM 集成】
检索本身不需要 LLM，但为了提升用户体验，通过 Function Calling 格式
定义了 4 个工具，让 LLM 能做查询扩展（"狗"→"puppy/canine/金毛"然后多路检索去重融合）
和结果解释（自然语言说明为什么返回这些图）。
LLM 没配置时自动降级为纯检索，不影响核心功能。
```

---

## 面试注意事项

### 不要做

- **不要把项目说成 RAG 系统**——你实现了检索+LLM 解释，但没有文档分块、混合检索、重排序等完整 RAG 能力。定位是"图像检索引擎 + LLM 辅助增强"
- **不要夸大**——压缩比就是 PQ 算法的数学属性（32 段 × 8bit = 32 bytes vs 512 × 4 = 2048 bytes），不是你调出来的
- **不要背代码**——理解设计思路比记住具体实现重要

### 要做

- **画架构图**——能徒手画出三层结构 + 双缓冲索引示意图
- **讲权衡**——为什么 buffer 用暴力 dot product 而不是也建 Faiss 索引（buffer < 1000 条，numpy 够快；超了就 merge）
- **讲为什么**——为什么 IVF add() 不需要 retrain（聚类中心和码本不变，新向量只需分配+编码）
- **诚实**——被问到"这是 RAG 吗"，直接说"是 RAG 中最基础的检索+生成部分，完整 RAG 还需要文档分块、混合检索、重排序等"

---

## 快速导航

| 你需要... | 去这里 |
|---|---|
| 快速了解项目亮点 | [核心亮点速览](#核心亮点速览) |
| 准备自我介绍 | [面试话术速记](#面试话术速记) |
| 理解系统架构 | [01-架构总览](./01-architecture-overview.md) |
| 深入三大模块 | [02-核心系统设计](./02-core-system-design.md) |
| 搞懂 IVF/PQ/双缓冲算法 | [03-关键算法实现](./03-key-algorithms.md) |
| 零基础理解概念 | [04-小白概念指南](./04-core-concepts-for-beginners.md) |
| 准备面试问答 | [05-面试准备指南](./05-interview-preparation.md) |

---

**最后更新**: 2026-06-01
**版本**: v2.0（对齐简历定稿）
