# 03 - 关键算法实现

> 📍 **阅读目标**: 理解核心算法的实现原理、代码细节和优化技巧

---

## 1. CLIP跨模态编码算法

### 1.1 CLIP原理概述

CLIP (Contrastive Language-Image Pre-training) 是OpenAI开发的多模态模型，通过对比学习将图像和文本映射到同一向量空间。

**核心思想**:

```
┌─────────────────────────────────────────────────────────────┐
│                      对比学习框架                            │
│                                                             │
│   图像编码器            文本编码器                           │
│   ┌─────────┐          ┌─────────┐                          │
│   │  Image  │          │  Text   │                          │
│   │    ↓    │          │    ↓    │                          │
│   │  ViT    │────┐     │Transformer│────┐                    │
│   │    ↓    │    │     │    ↓    │    │                    │
│   │512-dim  │    │     │512-dim  │    │                    │
│   └────┬────┘    │     └────┬────┘    │                    │
│        │         │          │         │                    │
│        └────┬────┘          └────┬────┘                    │
│             │                    │                          │
│             └──────→ 对比学习 ←───┘                          │
│                      (拉近正样本，推远负样本)                  │
└─────────────────────────────────────────────────────────────┘
```

**训练目标**: 最大化正样本对的余弦相似度，最小化负样本对的余弦相似度。

### 1.2 图像编码算法

**代码实现** (`retrieval_by_faiss.py:100-115`):

```python
def encode_image_by_path(self, path_img):
    """
    图像编码流程
    输入: 图像文件路径
    输出: 1×512 归一化特征向量
    """
    # Step 1: 读取图像 (OpenCV读取BGR格式)
    image_bgr = cv_imread(path_img)

    # Step 2: 颜色空间转换 (BGR → RGB)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Step 3: 转换为PIL Image
    image = Image.fromarray(image_rgb)

    # Step 4: CLIP预处理
    # 包括: Resize到224×224, 归一化, 标准化
    image_tensor = self.preprocess(image)  # (3, 224, 224)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Step 5: 模型推理
    with torch.no_grad():  # 关闭梯度计算，节省内存
        img_feat_vec = self.model.encode_image(image_tensor)  # (1, 512)

        # Step 6: L2归一化 (关键！)
        img_feat_vec /= img_feat_vec.norm(dim=-1, keepdim=True)

    # Step 7: 转换为numpy数组
    img_feat_vec = img_feat_vec.cpu().numpy()  # (1, 512)

    return img_feat_vec
```

**预处理流程详解**:

```python
# CLIP预处理包括:
1. Resize: 短边缩放到224，保持长宽比
2. CenterCrop: 中心裁剪224×224
3. ToTensor: 转换为[0,1]范围的tensor
4. Normalize: 使用ImageNet均值标准差
   mean=[0.48145466, 0.4578275, 0.40821073]
   std=[0.26862954, 0.26130258, 0.27577711]
```

### 1.3 文本编码算法

**代码实现** (`retrieval_by_faiss.py:134-145`):

```python
def encode_text_by_string(self, text):
    """
    文本编码流程
    输入: 文本字符串
    输出: 1×512 特征向量 (无需归一化)
    """
    # Step 1: Tokenize
    # CLIP使用BPE编码，最大长度77个token
    token = clip.tokenize([text]).to(self.device)  # (1, 77)

    # Step 2: 模型推理
    feat_text = self.model.encode_text(token)  # (1, 512)

    # Step 3: 转换为numpy (注意: 不归一化！)
    feat_text = feat_text.detach().cpu().numpy()

    return feat_text
```

**Tokenization说明**:

```
输入文本: "a dog playing"
    │
    ▼
BPE编码:
- "a" → token_id: 320
- "dog" → [subword1, subword2]
- "playing" → [subword1, subword2]

输出: [SOT] token1 token2 ... [EOT] [PAD]...
      │←─ 实际文本 ─→│
      │←──────── 77 ────────→│

SOT = Start Of Text
EOT = End Of Text
PAD = 填充token
```

### 1.4 L2归一化的数学原理

**为什么需要归一化？**

```
原始CLIP输出未归一化，但我们需要计算余弦相似度:

cos_sim(A, B) = (A·B) / (||A|| × ||B||)

L2归一化: A' = A / ||A||
         则 ||A'|| = 1

归一化后:
cos_sim(A', B') = A'·B'  (因为||A'||=||B'||=1)

Faiss使用L2距离:
||A' - B'||² = ||A'||² + ||B'||² - 2(A'·B')
             = 1 + 1 - 2cos_sim
             = 2(1 - cos_sim)

结论: L2距离越小 ↔ 余弦相似度越大 ↔ 越相似
```

**代码验证**:

```python
import numpy as np

# 两个向量
a = np.array([3.0, 4.0])
b = np.array([6.0, 8.0])

# 余弦相似度 (未归一化)
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# = 50 / (5 * 10) = 1.0 (完全同向)

# L2归一化
a_norm = a / np.linalg.norm(a)  # [0.6, 0.8]
b_norm = b / np.linalg.norm(b)  # [0.6, 0.8]

# 归一化后的内积 = 余弦相似度
dot_product = np.dot(a_norm, b_norm)  # 1.0

# L2距离
l2_dist = np.linalg.norm(a_norm - b_norm)  # 0.0
```

---

## 2. Faiss向量检索算法

### 2.1 IVF (倒排文件索引)

**原理**:

```
┌─────────────────────────────────────────────────────────────┐
│                    IVF索引结构                              │
│                                                             │
│   步骤1: K-Means聚类训练                                     │
│   ┌─────────────────────────────────────────┐               │
│   │    所有向量 (N×512)                     │               │
│   │         ↓                               │               │
│   │    K-Means (k=4096)                     │               │
│   │         ↓                               │               │
│   │    4096个聚类中心 (4096×512)            │               │
│   └─────────────────────────────────────────┘               │
│                                                             │
│   步骤2: 向量分配到最近中心                                  │
│   ┌─────────┐  ┌─────────┐        ┌─────────┐              │
│   │  中心0  │  │  中心1  │  ...   │ 中心4095│              │
│   │  列表   │  │  列表   │        │  列表   │              │
│   │ ┌───┐  │  │ ┌───┐  │        │ ┌───┐  │              │
│   │ │v1 │  │  │ │v3 │  │        │ │v2 │  │              │
│   │ │v5 │  │  │ │v7 │  │        │ │v4 │  │              │
│   │ └───┘  │  │ └───┘  │        │ └───┘  │              │
│   └─────────┘  └─────────┘        └─────────┘              │
│                                                             │
│   步骤3: 查询时只搜索nprobe个最近列表                        │
└─────────────────────────────────────────────────────────────┘
```

**参数说明**:

```python
index_string = "IVF4096"
# IVF: 倒排文件 (Inverted File)
# 4096: 聚类中心数量 (nlist)

# 查询参数
nprobe = 10  # 搜索最近的10个聚类列表
# 默认nprobe=1，精度低但速度快
# 增大nprobe提高精度，但速度变慢
```

### 2.2 PQ (乘积量化)

**原理**:

```
┌─────────────────────────────────────────────────────────────┐
│                   乘积量化 (PQ)                              │
│                                                             │
│   原始向量: 512维 float32 (2048字节)                         │
│        ↓                                                    │
│   分割成m=32个子向量                                         │
│   ┌────┬────┬────┬────┬────┬────┐                          │
│   │16d │16d │16d │16d │... │16d │  每个子向量16维            │
│   └────┴────┴────┴────┴────┴────┘                          │
│     │    │    │    │         │                             │
│     ↓    ↓    ↓    ↓         ↓                             │
│   每个子向量独立量化                                          │
│                                                             │
│   码本学习: 每个子空间训练256个中心 (k=256, 8bit)            │
│   ┌─────────┐  ┌─────────┐        ┌─────────┐              │
│   │  码本0  │  │  码本1  │  ...   │  码本31 │              │
│   │256×16d │  │256×16d │        │256×16d │              │
│   └─────────┘  └─────────┘        └─────────┘              │
│                                                             │
│   编码后: 32个code (每个8bit) = 32字节                        │
│   压缩比: 2048/32 = 64x                                     │
└─────────────────────────────────────────────────────────────┘
```

**距离计算**:

```
查询向量 q = [q1, q2, ..., q32]  (每个qi是16维)

距离计算:
1. 查询向量也分割成32个子向量
2. 对每个子向量qi，计算与码本中256个中心的距离
3. 得到32×256的查找表 (ADC表)
4. 量化向量v的距离 = sum(查找表[i][code_i])

复杂度: O(32) 每次距离计算
vs Flat: O(512) 每次距离计算
```

### 2.3 IVF-PQ组合索引

**代码实现** (`retrieval_by_faiss.py:47-73`):

```python
def _init_index(self, feat_mat):
    """
    初始化Faiss索引 (IVF4096,PQ32x8)
    """
    # 数据类型检查
    if feat_mat.dtype != np.float32:
        feat_mat = feat_mat.astype(np.float32)

    # 使用工厂方法创建索引
    # IVF4096: 4096个倒排列表
    # PQ32x8: 32个子空间，每个8bit (256个中心)
    index = faiss.index_factory(self.feat_dim, self.index_string)

    # GPU加速 (如果可用)
    if USE_GPU and res is not None:
        self.index = faiss.index_cpu_to_gpu(res, gpu_id, index)
        print('[IndexModule] 使用GPU索引')
    else:
        self.index = index
        print('[IndexModule] 使用CPU索引')

    # 训练索引 (学习聚类中心和码本)
    s1 = time.time()
    self.index.train(feat_mat)
    s2 = time.time()

    # 添加数据
    self.index.add(feat_mat)

    print(f'training time:{s2-s1}s, train set:{feat_mat.shape}')
```

**参数调优指南**:

| 参数 | 含义 | 建议值 |
|:---|:---|:---|
| nlist (IVF) | 聚类中心数 | 4×√N ~ 16×√N |
| m (PQ) | 子向量数 | 维数/2 ~ 维数/4 |
| nbits (PQ) | 每子空间码数 | 8 (256中心) 或 16 (65536中心) |
| nprobe | 查询时搜索的列表数 | 1~100，默认1 |

**示例计算**:

```python
# 100万数据，512维
N = 1_000_000
dim = 512

# IVF参数
nlist = 4096  # 约4×√N = 4000

# PQ参数
m = 32  # 512/32 = 16维每子空间
nbits = 8  # 256个中心

# 内存计算
# 原始: 1M × 512 × 4B = 2GB
# PQ后: 1M × 32 × 1B = 32MB (压缩比 64x)
# IVF开销: 4096个列表指针 ≈ 可忽略

# 搜索复杂度
# Flat: 1M次距离计算
# IVF-PQ (nprobe=10): 10个列表 × (1M/4096) ≈ 2500次距离计算
# 加速比: ~400x
```

---

## 3. RAG检索流程算法

### 3.1 查询扩展算法

**算法流程**:

```
输入: user_query (字符串), num_expansions (扩展数量)
输出: expanded_queries (查询列表)

┌─────────────────────────────────────────────────────────────┐
│ 查询扩展算法                                                 │
│                                                             │
│ 1. 构建Prompt                                               │
│    - 角色定义: 图像搜索助手                                  │
│    - 任务: 生成N个不同表达的搜索意图                          │
│    - 约束: 保持语义、不同角度、简洁                           │
│                                                             │
│ 2. 调用LLM                                                  │
│    - temperature=0.7 (有一定创造性)                          │
│    - max_tokens=500                                         │
│                                                             │
│ 3. 后处理                                                   │
│    - 按行分割                                                │
│    - 过滤空行                                                │
│    - 去重                                                    │
│    - 确保原始查询在首位                                      │
│    - 截断到num_expansions+1个                               │
└─────────────────────────────────────────────────────────────┘
```

**代码实现** (`llm_interface.py:60-111`):

```python
def expand_query(self, user_query: str, num_expansions: int = 3) -> List[str]:
    # 降级处理
    if not self._available:
        return [user_query]

    # Prompt工程
    prompt = f"""作为图像搜索助手，请基于用户的查询生成{num_expansions}个不同表达的搜索意图，以提高检索召回率。

用户查询: {user_query}

要求:
1. 保持原始语义，但使用不同的词汇和表达方式
2. 可以从不同角度描述（如风格、场景、对象、颜色等）
3. 每个扩展查询简洁明了，不超过20个字
4. 直接返回扩展查询列表，每行一个，不要编号

输出:"""

    try:
        # LLM调用
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个专业的图像搜索查询扩展助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # 解析响应
        expanded = response.choices[0].message.content.strip().split('\n')

        # 后处理
        expanded = [q.strip() for q in expanded if q.strip()]  # 过滤空行
        expanded = list(dict.fromkeys(expanded))  # 去重保持顺序

        # 确保包含原始查询
        if user_query not in expanded:
            expanded.insert(0, user_query)

        return expanded[:num_expansions + 1]

    except Exception as e:
        print(f"[LLMInterface] 查询扩展失败: {e}")
        return [user_query]  # 失败时降级
```

**示例**:

```
输入: "a dog playing in the park"

LLM输出:
a dog playing in the park
dog running on grass field
puppy playing outdoors
pet enjoying the garden
canine having fun outside

后处理后:
["a dog playing in the park", "dog running on grass field",
 "puppy playing outdoors", "pet enjoying the garden"]
```

### 3.2 多查询融合算法

**算法流程** (`rag_engine.py:93-118`):

```python
def _merge_multi_query_results(self, query_results_list, topk):
    """
    融合多查询结果

    策略:
    1. 使用Set去重，确保同一图片只出现一次
    2. 保留每个图片的最短距离（最相似）
    3. 按距离全局排序
    4. 截断取topk
    """
    all_results = []
    seen_paths = set()

    for query, (distance_result, index_result, path_list) in query_results_list:
        for dist, idx, path in zip(distance_result, index_result, path_list):
            if path not in seen_paths and path != 'None':
                seen_paths.add(path)
                all_results.append({
                    "path": path,
                    "distance": float(dist),
                    "index": int(idx),
                    "matched_query": query
                })

    # 全局排序（按距离升序）
    all_results.sort(key=lambda x: x["distance"])

    # 截断
    return all_results[:topk]
```

**融合策略说明**:

```
查询1结果: [(img_a, 0.1), (img_b, 0.2), (img_c, 0.3)]
查询2结果: [(img_b, 0.15), (img_d, 0.25)]  # img_b重复

融合过程:
1. 遍历查询1: 添加img_a(0.1), img_b(0.2), img_c(0.3)
2. 遍历查询2: img_b已存在(0.2 < 0.15)，跳过
   添加img_d(0.25)
3. 排序: img_a(0.1), img_b(0.2), img_c(0.3), img_d(0.25)
4. 取top3: img_a, img_b, img_d

注意: 实际代码中不比较距离，直接跳过重复项
(依赖Faiss返回的已排序结果)
```

### 3.3 完整RAG检索流程

```
┌─────────────────────────────────────────────────────────────┐
│              RAG检索算法 (search_and_explain)               │
│                                                             │
│  输入: query, topk=10, use_expansion=True                   │
│  输出: {original_query, expanded_queries, results,          │
│         ai_explanation, total_results}                      │
│                                                             │
│  Step 1: 查询扩展                                           │
│  ─────────────────                                          │
│  IF use_expansion AND enable_expansion AND 是文本查询:      │
│      expanded = llm.expand_query(query)                     │
│  ELSE:                                                      │
│      expanded = [query]                                     │
│                                                             │
│  Step 2: 多查询检索                                          │
│  ─────────────────                                          │
│  all_results = []                                           │
│  seen_paths = Set()                                         │
│  FOR each q IN expanded:                                    │
│      distance, ids, paths = retrieval_module.retrieval(q)   │
│      FOR dist, idx, path IN zip(distance, ids, paths):      │
│          IF path NOT IN seen_paths:                         │
│              seen_paths.add(path)                           │
│              all_results.append({path, distance, idx, q})   │
│                                                             │
│  Step 3: 重排序                                              │
│  ─────────────────                                          │
│  SORT all_results BY distance ASC                           │
│  final_results = all_results[:topk]                         │
│                                                             │
│  Step 4: AI解释 (可选)                                       │
│  ─────────────────                                          │
│  IF enable_explanation:                                     │
│      explanation = llm.explain_results(query, final)        │
│                                                             │
│  RETURN result_dict                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Agent工具调用算法

### 4.1 OpenAI Function Calling格式

**工具定义格式**:

```python
{
    "type": "function",
    "function": {
        "name": "search_by_text",
        "description": "根据文本描述搜索相关图片",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索文本描述"
                },
                "topk": {
                    "type": "integer",
                    "description": "返回结果数量"
                }
            },
            "required": ["query"]
        }
    }
}
```

### 4.2 工具选择算法

```
┌─────────────────────────────────────────────────────────────┐
│              LLM工具选择流程                                 │
│                                                             │
│  系统提示 (System Prompt):                                  │
│  ───────────────────────                                    │
│  你是一个智能图像检索助手。你可以使用以下工具:              │
│                                                             │
│  - search_by_text: 根据文本描述搜索相关图片                 │
│    参数: query(string, required), topk(integer)             │
│                                                             │
│  - search_by_image: 根据图片搜索相似图片                    │
│    参数: image_path(string, required), topk(integer)        │
│                                                             │
│  请根据用户需求选择合适的工具。                             │
│  直接以JSON格式返回工具调用:                                │
│  {"tool": "tool_name", "arguments": {"arg1": "value1"}}     │
│                                                             │
│  用户输入: "帮我找几张小狗的图片"                            │
│      │                                                      │
│      ▼                                                      │
│  LLM推理                                                    │
│      │                                                      │
│      ▼                                                      │
│  {"tool": "search_by_text",                                  │
│   "arguments": {"query": "puppy dog", "topk": 5}}            │
│                                                             │
│  解析 → 执行 → 返回结果                                     │
└─────────────────────────────────────────────────────────────┘
```

**代码实现** (`agent_tools.py:354-410`):

```python
def run(self, user_input: str) -> Dict[str, Any]:
    # 构建系统提示
    system_prompt = f"""你是一个智能图像检索助手。你可以使用以下工具帮助用户:

{self.tools.get_tools_description()}

请根据用户需求选择合适的工具。如果需要多个步骤，请逐步执行。
直接以JSON格式返回工具调用:
{{"tool": "tool_name", "arguments": {{"arg1": "value1"}}}}"""

    # 调用LLM
    response = self.llm.client.chat.completions.create(
        model=self.llm.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.3,  # 低温度，确定性输出
        max_tokens=500
    )

    content = response.choices[0].message.content

    # 解析JSON
    try:
        # 尝试从代码块中提取
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content.strip()

        tool_call = json.loads(json_str)
        tool_name = tool_call.get("tool")
        arguments = tool_call.get("arguments", {})

        # 执行工具
        result = self.tools.execute_tool(tool_name, arguments)

        return {
            "user_input": user_input,
            "tool_call": tool_call,
            "result": result,
            "success": True
        }

    except json.JSONDecodeError:
        # 不是有效工具调用，返回直接回复
        return {
            "user_input": user_input,
            "direct_response": content,
            "tool_call": None
        }
```

---

## 5. 性能优化技巧

### 5.1 批处理优化

```python
# 当前实现: 单查询
for q in expanded_queries:
    result = retrieval_module.retrieval_func(q, topk)

# 优化方案: 批处理编码 + 批处理检索
# 1. 批量编码所有查询文本
query_features = clip_model.encode_text_batch(expanded_queries)

# 2. 批量检索 (如果Faiss支持)
all_distances, all_ids = index.search_batch(query_features, topk)
```

### 5.2 缓存优化

```python
# 查询结果缓存
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_retrieval(query_hash):
    """缓存热门查询结果"""
    return retrieval_module.retrieval_func(query, topk)

# 特征预计算
# 启动时预加载所有特征到内存，避免IO
with open(CFG.feat_mat_path, 'rb') as f:
    feat_mat = pickle.load(f)  # 一次性加载
```

### 5.3 索引参数调优

```python
# 根据数据量调整nlist
import math

def optimal_nlist(n_vectors):
    """计算最优nlist值"""
    return min(4 * int(math.sqrt(n_vectors)), n_vectors // 39)

# 查询时调整nprobe平衡精度和速度
index.nprobe = 10  # 默认值
# 高精度场景
index.nprobe = 50
# 高性能场景
index.nprobe = 1
```

---

## 6. 增量索引算法

### 6.1 增量添加算法

**问题背景**: Faiss IVF索引训练后如何添加新向量？

**解决方案**: 缓冲区模式

```
算法: 增量添加
─────────────────────────────────────────
输入: image_paths (新图片路径列表)
输出: added_ids (新分配的ID列表)

1. 特征提取
   features, valid_paths = index_builder.extract_features(image_paths)

2. 分配ID
   start_id = next_id
   new_ids = [start_id, start_id + 1, ..., start_id + len(features) - 1]
   next_id += len(features)

3. 存入缓冲区
   for feat, path, id in zip(features, valid_paths, new_ids):
       buffer_features.append(feat)
       buffer_paths.append(path)
       buffer_ids.append(id)
       id_to_path[id] = path

4. 检查阈值
   if len(buffer_ids) >= buffer_size_threshold:
       提示: "缓冲区已满，建议执行 merge_buffer()"

5. 返回结果
   return {"added": len(valid_paths), "ids": new_ids}
```

**复杂度分析**:

| 步骤 | 时间复杂度 | 说明 |
|:---|:---|:---|
| 特征提取 | O(N × image_decode × CLIP) | N=图片数，每张约100ms |
| ID分配 | O(N) | 简单范围生成 |
| 存入缓冲区 | O(N) | 列表append操作 |
| **总计** | **O(N × CLIP编码)** | 主要开销在特征提取 |

### 6.2 逻辑删除算法

**问题背景**: Faiss IVF不支持直接删除向量

**解决方案**: 标记删除 + 搜索过滤

```
算法: 逻辑删除
─────────────────────────────────────────
输入: image_ids (要删除的ID列表)
输出: remove_count (成功删除数量)

1. 初始化计数器
   removed = 0

2. 遍历每个ID
   for id in image_ids:
       # 验证ID存在性
       exists_in_main = (id < main_index.ntotal)
       exists_in_buffer = (id in buffer_ids)

       if not exists_in_main and not exists_in_buffer:
           continue  # ID不存在，跳过

       # 逻辑删除：添加到删除集合
       deleted_ids.add(id)
       removed += 1

       # 缓冲区优化：立即移除
       if id in buffer_ids:
           idx = buffer_ids.index(id)
           buffer_features.pop(idx)
           buffer_paths.pop(idx)
           buffer_ids.pop(idx)

3. 返回结果
   return {"removed": removed}
```

**为什么不用物理删除？**

```
Faiss IVF索引结构:
┌─────────────────────────────────────────┐
│  聚类中心列表 (4096个)                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ 中心0   │ │ 中心1   │ │ 中心2   │   │
│  │ [v1,v5] │ │ [v2,v3] │ │ [v4,v6] │   │
│  └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────┘

问题: 向量v3分散存储在倒排列表中
删除v3需要:
1. 找到v3属于哪个中心 → 需要倒排映射
2. 从对应列表中移除 → 需要列表操作
3. 更新索引结构 → 可能触发重建

Faiss不提供这些操作，因为:
- 设计初衷是静态索引
- 动态删除会严重影响IVF结构
- 删除后聚类中心可能需要重新计算

解决方案: 逻辑删除
- deleted_ids集合存储已删除ID
- 搜索时过滤 (O(1)哈希查找)
- 定期重建清理 (批量处理)
```

### 6.3 双路搜索合并算法

**问题背景**: 搜索时需要同时查主索引和缓冲区

**算法流程**:

```python
def search(query_vector, topk):
    """
    双路搜索合并算法
    """
    # 1. 搜索主索引 (Faiss IVF-PQ)
    main_dist, main_ids = main_index.search(query_vector, topk * 2)
    # 复杂度: O(nprobe × (N/nlist) × m)

    # 2. 搜索缓冲区 (暴力搜索)
    buffer_matrix = np.vstack(buffer_features)  # (M, 512)
    similarities = np.dot(buffer_matrix, query_vector.T).flatten()
    buffer_dist = 1 - similarities
    top_indices = np.argsort(buffer_dist)[:topk * 2]
    # 复杂度: O(M × d)  M=缓冲区大小

    # 3. 合并结果
    all_results = []

    # 添加主索引结果 (过滤已删除)
    for dist, id in zip(main_dist, main_ids):
        if id not in deleted_ids:      # O(1) 哈希查找
            all_results.append((dist, id))

    # 添加缓冲区结果 (过滤已删除)
    for idx in top_indices:
        id = buffer_ids[idx]
        if id not in deleted_ids:      # O(1) 哈希查找
            all_results.append((buffer_dist[idx], id))

    # 4. 全局排序取topk
    all_results.sort(key=lambda x: x[0])  # O((K+M) log(K+M))
    return all_results[:topk]
```

**复杂度对比**:

| 搜索类型 | 时间复杂度 | 当M=1000, N=100K时 |
|:---|:---|:---|
| 仅主索引 | O(nprobe×(N/nlist)×m) | ~10K 操作 |
| 仅缓冲区 | O(M×d) | ~512K 操作 |
| **双路合并** | **O(nprobe×(N/nlist)×m + M×d)** | **~522K 操作** |

**注意**: 缓冲区通常很小(M<1000)，所以开销可接受

### 6.4 缓冲区合并算法

**触发条件**:
1. 缓冲区大小超过阈值 (buffer_size_threshold)
2. 手动调用 merge_buffer()
3. 服务关闭前

```
算法: 缓冲区合并
─────────────────────────────────────────
输入: 无 (使用当前缓冲区)
输出: merged_count (合并的向量数)

1. 检查缓冲区
   if len(buffer_features) == 0:
       return {"merged": 0}

2. 合并特征
   buffer_matrix = np.vstack(buffer_features)  # (M, 512)

3. 添加到主索引
   # Faiss IVF支持add操作!
   new_ids = main_index.add_vectors(buffer_matrix)

4. 清空缓冲区
   buffer_features = []
   buffer_paths = []
   buffer_ids = []

5. 返回结果
   return {"merged": M, "total": main_index.ntotal}
```

**为什么Faiss IVF支持add但不支持delete？**

```
IVF索引结构训练后:
┌─────────────────────────────────────────┐
│  聚类中心 (固定)                        │
│  [c1, c2, ..., c4096]                   │
│                                         │
│  倒排列表                               │
│  list_1: [v1, v5, ...]                  │
│  list_2: [v2, v3, ...]                  │
│  ...                                    │
└─────────────────────────────────────────┘

添加新向量v_new:
1. 计算v_new与所有聚类中心的距离
2. 找到最近的k个中心 (通常是1个)
3. 将v_new添加到对应倒排列表
4. 无需重新训练！

删除向量v_old:
1. 需要知道v_old在哪个倒排列表
2. 需要从列表中移除 (列表是数组，移除需移动元素)
3. 删除后聚类中心可能偏移 (因为分布变了)
4. 可能需要重新训练

结论: add简单，delete复杂
```

### 6.5 索引重建算法

**触发条件**:
1. 已删除数据积累过多 (deleted_ids > threshold)
2. 需要优化索引结构
3. 定期维护 (如每周)

```
算法: 索引重建
─────────────────────────────────────────
输入: all_features, all_paths, index_string
输出: new_index (重建后的索引)

1. 过滤已删除数据
   valid_indices = [i for i in range(len(all_paths))
                   if i not in deleted_ids]
   valid_features = all_features[valid_indices]
   valid_paths = [all_paths[i] for i in valid_indices]

2. 创建新索引
   new_index = IndexModule(index_string, feat_dim, valid_features)

3. 清空状态
   deleted_ids.clear()
   buffer_features = []
   buffer_paths = []
   buffer_ids = []
   next_id = new_index.ntotal

4. 更新映射
   id_to_path = {i: path for i, path in enumerate(valid_paths)}

5. 返回新索引
   return new_index
```

**重建成本**:

| 数据量 | 重建时间 | 内存需求 |
|:---|:---|:---|
| 10K | ~10秒 | ~100MB |
| 100K | ~2分钟 | ~500MB |
| 1M | ~30分钟 | ~2GB |

**建议**: 在低峰期执行重建，或采用后台线程

---

## 7. 算法复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 | 备注 |
|:---|:---|:---|:---|
| CLIP编码 | O(1) | O(512) | 单次前向传播 |
| Faiss Flat | O(N×d) | O(N×d) | N=向量数, d=维度 |
| Faiss IVF | O(nprobe×(N/nlist)×d) | O(N×d) | nlist=聚类数 |
| Faiss PQ | O(m×2^nbits + m) | O(N×m) | m=子向量数 |
| Faiss IVF-PQ | O(nprobe×(N/nlist)×m) | O(N×m) | 组合优势 |
| **增量添加** | **O(M×CLIP)** | **O(M×d)** | M=添加数量 |
| **逻辑删除** | **O(M)** | **O(D)** | D=已删除数 |
| **双路搜索** | **O(nprobe×(N/nlist)×m + B×d)** | **O(K)** | B=缓冲区大小 |
| **缓冲区合并** | **O(M×log(N))** | **O(M×d)** | M=缓冲区大小 |
| 查询扩展 | O(LLM延迟) | O(k) | k=扩展数 |
| 多查询融合 | O(k×topk×log(k×topk)) | O(k×topk) | 排序复杂度 |

**实际数值** (N=120K, d=512, nlist=4096, nprobe=10, m=32, B=1000):

```
Flat: 120K × 512 = 61M 次操作
IVF-PQ: 10 × (120K/4096) × 32 ≈ 10K 次操作
加速比: ~6000x

增量索引场景:
- 主索引搜索: ~10K 次操作
- 缓冲区搜索 (B=1000): 1000 × 512 = 512K 次操作
- 双路合并: ~522K 次操作
- 相比Flat仍有 ~100x 加速
```

---

**上一章**: [02 - 核心系统设计](./02-core-system-design.md)

**下一章**: [04 - 小白概念指南](./04-core-concepts-for-beginners.md) - 用大白话理解技术概念
