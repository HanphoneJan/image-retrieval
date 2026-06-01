# 03 - 关键算法实现

> **阅读目标**: 理解 IVF 倒排、PQ 量化、双缓冲搜索合并、增量索引的算法原理

---

## 1. IVF 倒排索引

### 1.1 原理

```
┌────────────────────────────────────────────────────────┐
│                  IVF 索引结构                           │
│                                                        │
│  训练阶段: 全量向量 → KMeans(k=4096) → 4096个聚类中心   │
│                                                        │
│  ┌─────────┐  ┌─────────┐        ┌─────────┐          │
│  │ 中心 0  │  │ 中心 1  │  ...   │中心 4095│          │
│  │ [v1,v5] │  │ [v3,v7] │        │ [v2,v4] │          │
│  └─────────┘  └─────────┘        └─────────┘          │
│                                                        │
│  查询时: query → 找最近 nprobe 个中心 → 只搜这些列表    │
│                                                        │
│  复杂度: O(nprobe × N/nlist) vs Flat O(N)              │
│  nprobe=10, N=12万: ~300 次距离计算 vs 12万次          │
└────────────────────────────────────────────────────────┘
```

### 1.2 参数选择

`nlist = 4096` 的选择依据:

```python
# 经验公式: nlist = 4 × sqrt(N) ~ 16 × sqrt(N)
N = 120_000
nlist_min = 4 * int(N ** 0.5)   # ≈ 1386
nlist_max = 16 * int(N ** 0.5)  # ≈ 5543
# 取 Faiss 推荐的 2 的幂: 4096
```

---

## 2. PQ 乘积量化

### 2.1 原理

```
原始向量: 512 维 float32 = 2048 bytes

Step 1: 切分
[0.1, 0.5, 0.8, ..., 0.3]  (512个float32)
  → 切成 32 段, 每段 16 维

Step 2: 每段独立训练码本
段 0 的 16 维 → KMeans(k=256) → 256 个中心 → 8bit 编码
段 1 的 16 维 → KMeans(k=256) → 256 个中心 → 8bit 编码
...
段 31 同理

Step 3: 编码
对每个向量, 每段找最近的码本中心 → 存编码值(0-255)
压缩后: 32 段 × 1 byte = 32 bytes
压缩比: 2048 / 32 = 64x
```

### 2.2 距离计算 (ADC — Asymmetric Distance Computation)

```
查询向量 q = [q0, q1, ..., q31]  (不量化, 保持精度)

Step 1: 预计算查询与所有码本中心的距离
  对每段 i: 计算 qi 与该段 256 个中心的距离 → 查找表 (32 × 256)

Step 2: 对每个数据库向量 v
  v 已编码为 [code_0, code_1, ..., code_31]
  dist(q, v) = sum(查找表[i][code_i])  for i in 0..31

复杂度: O(32) 查表求和, vs Flat O(512) 浮点乘法
```

### 2.3 PQ 基准测试结论

来自 [demos/02_pq_benchmark.py](../../demos/02_pq_benchmark.py) 在 SIFT1M 上的实验:

| 配置 | Recall@1 | Recall@10 | Recall@100 | 每向量字节 |
|:---|:---|:---|:---|:---|
| PQ4x8 | ~0.15 | ~0.40 | ~0.70 | 4 |
| PQ8x8 | ~0.25 | ~0.55 | ~0.85 | 8 |
| PQ16x8 | ~0.35 | ~0.70 | ~0.93 | 16 |
| **PQ32x8** | ~0.40 | ~0.78 | ~0.96 | **32** |

8bit 量化在 CPU 上有特殊优化 (SSE), 速度快于其他 bit 数。16 子段是 sweet spot, 32 子段精度略高但时间翻倍。

---

## 3. IVF+PQ 组合

### 3.1 两级检索

```
查询向量 q
  │
  ▼
IVF 粗排: q 和 4096 个聚类中心算距离 → 取最近 nprobe 个
  │
  ▼
PQ 精排: 在每个选中列表内, 用 ADC 查表计算与各向量的近似距离
  │
  ▼
合并所有列表的结果 → 排序 → topk
```

### 3.2 基准测试结论

来自 [demos/03_ivfpq_benchmark.py](../../demos/03_ivfpq_benchmark.py):

```
IVF+PQ32x8 在 SIFT1M 上:
  nprobe=1:  Recall@100 ≈ 0.20, ~1ms
  nprobe=4:  Recall@100 ≈ 0.55, ~3ms
  nprobe=16: Recall@100 ≈ 0.85, ~8ms
  nprobe=64: Recall@100 ≈ 0.95, ~20ms

中心数越大: 每列表向量越少 → 搜索越快, 但 4096 后 recall 趋于稳定
```

这直接指导了 `IVF4096,PQ32x8` 的选型: 中心数够大保证搜索速度, PQ32x8 保证精度可调。

---

## 4. 双缓冲增量索引算法

### 4.1 add 算法

```
算法: 增量添加
─────────────────
输入: image_paths (新图片路径列表)
输出: added_ids

1. 特征提取
   features, valid_paths = index_builder.extract_features(image_paths)

2. 分配 ID
   new_ids = [next_id, next_id+1, ..., next_id+len(features)-1]
   next_id += len(features)

3. 存入缓冲区
   for feat, path, id in zip(features, valid_paths, new_ids):
       buffer_features.append(feat.reshape(1, -1))
       buffer_paths.append(path)
       buffer_ids.append(id)
       id_to_path[id] = path

4. 检查阈值
   if len(buffer_ids) >= buffer_size_threshold:
       print("建议执行 merge_buffer()")

5. 返回
   return {"added": len(valid_paths), "ids": new_ids}
```

### 4.2 remove 算法 (逻辑删除)

```
算法: 逻辑删除
─────────────────
输入: image_ids (要删除的 ID 列表)
输出: {removed, not_found}

for id in image_ids:
    exists = (id < main_index.ntotal) or (id in buffer_ids)

    if not exists:
        not_found.append(id)
        continue

    deleted_ids.add(id)        # 逻辑标记, O(1)
    removed += 1

    if id in buffer_ids:       # 在缓冲区 → 立即物理移除
        idx = buffer_ids.index(id)
        buffer_features.pop(idx)
        buffer_paths.pop(idx)
        buffer_ids.pop(idx)

return {"removed": removed, "not_found": not_found}
```

为什么对主索引只做逻辑删除、对 buffer 做物理删除?

```
主索引: Faiss IVF 不支持逐条 delete
         → 标记后搜索时 O(1) 哈希过滤
         → 删除积累多了 → rebuild 彻底清理

缓冲区: Python list, pop 就行
         → 立即物理移除, 零代价
```

### 4.3 search 算法 (双路合并)

```
算法: 双路搜索合并
─────────────────
输入: query_vector (1, 512), topk
输出: (distances, ids, paths)

1. 搜索主索引
   main_dist, main_ids = main_index.feat_retrieval(query_vector, topk*2)
   # topk*2 给去重预留空间

2. 搜索缓冲区 (暴力 dot product)
   buffer_matrix = np.vstack(buffer_features)   # (M, 512)
   similarities = np.dot(buffer_matrix, query_vector.T).flatten()  # (M,)
   # 向量已 L2 归一化, dot product = 余弦相似度
   distances = 1 - similarities   # 转为距离
   top_indices = np.argsort(distances)[:topk*2]

3. 合并 + 过滤删除
   all_results = []
   for dist, id in zip(main_dist, main_ids):
       if id not in deleted_ids:        # O(1)
           all_results.append((dist, id, get_path(id)))
   for idx in top_indices:
       id = buffer_ids[idx]
       if id not in deleted_ids:
           all_results.append((distances[idx], id, buffer_paths[idx]))

4. 全局排序 → 取 topk
   all_results.sort(key=lambda x: x[0])
   return all_results[:topk]
```

复杂度分析:

```
主索引: O(nprobe × N/nlist × 32)  — Faiss IVF+PQ
缓冲区: O(M × 512)                 — numpy dot product
合并:   O((topk*2 + topk*2) log(topk*4))

当 N=12万, M=500, nprobe=10:
主索引: ~10 × 30 × 32 ≈ 9600 次操作
缓冲区: 500 × 512 = 256K 次操作
→ 相比 Flat 的 120K × 512 = 61M 次操作, 仍有 ~200x 加速
```

### 4.4 add_vectors 免训原理

```python
# retrieval_by_faiss.py:89-110
def add_vectors(self, vectors):
    start_id = self.index.ntotal
    self.index.add(vectors)  # 直接调 Faiss 原生 add, 不 retrain
    return list(range(start_id, start_id + len(vectors)))
```

为什么 Faiss IVF 支持 add 但不支持 delete?

```
支持 add:
  聚类中心和码本在 train 阶段已固定
  新向量只需:
    1. 找最近聚类中心 → 放入倒排列表    (不需要重新聚类)
    2. 用已有码本编码                    (不需要重新训练码本)

不支持 delete:
  1. 需知道向量在哪个倒排列表 (无反向索引)
  2. 从列表中移除需要移动数组元素
  3. 删除后聚类中心可能偏移
```

但代价是: 新数据分布变了, 编码质量下降 → 定期 rebuild 用全量数据重新 train。

---

## 5. LLM 查询扩展算法

```
算法: LLM 查询扩展
─────────────────
输入: user_query, num_expansions=3
输出: expanded_queries (包含原始查询)

1. 不可用 → 降级返回 [user_query]

2. 构建 Prompt (角色+任务+约束+输出格式)

3. LLM 调用 (temperature=0.7, 有一定创造性)

4. 后处理:
   - 按行 split
   - 过滤空行
   - dict.fromkeys 去重 (保持顺序)
   - 确保原始查询在首位
   - 截断到 num_expansions+1

5. 异常 → 降级返回 [user_query]
```

**扩展示例**:
```
输入: "a dog playing in the park"

LLM 输出:
a dog playing in the park
dog running on grass field
puppy playing outdoors
pet enjoying the garden

→ 4 个查询分别检索 → 合并去重 → 提升覆盖范围
```

---

## 6. Function Calling 工具调用算法

```
用户输入 "帮我找几张小狗的图片"
  │
  ▼
LLM + System Prompt (包含 4 个工具描述 + JSON 格式要求)
  │
  ▼
LLM 输出: {"tool": "search_by_text", "arguments": {"query": "小狗", "topk": 5}}
  │
  ▼
解析 JSON → extract tool_name + arguments
  │
  ▼
execute_tool("search_by_text", {"query": "小狗", "topk": 5})
  → func(**arguments)  # 实际执行
  → 返回 {tool, arguments, result, success}
  │
  ▼
结果返回给用户
```

LLM 负责决策 (调用哪个工具、传什么参数), 系统负责执行。这是 OpenAI Function Calling 的标准范式。

---

**上一章**: [02 - 核心系统设计](./02-core-system-design.md)
**下一章**: [04 - 小白概念指南](./04-core-concepts-for-beginners.md) — 大白话理解核心概念
