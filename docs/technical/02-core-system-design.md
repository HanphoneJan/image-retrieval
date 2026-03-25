# 02 - 核心系统设计

> 📍 **阅读目标**: 深入理解每个核心模块的设计思路、接口定义和实现细节

---

## 1. 检索层设计

### 1.1 CLIPModel - 多模态编码器

**职责**: 统一编码图像和文本到512维向量空间

**类定义** (`retrieval_by_faiss.py:91-146`):

```python
class CLIPModel:
    def __init__(self, clip_backbone_type, device):
        self.model, self.preprocess = clip.load(
            clip_backbone_type,
            device=device,
            jit=False
        )

    def encode_image_by_path(self, path_img: str) -> np.ndarray
    def encode_image_by_ndarray(self, image_rgb: np.ndarray) -> np.ndarray
    def encode_text_by_string(self, text: str) -> np.ndarray
```

**图像编码流程**:

```
图像路径
    │
    ▼
┌───────────────┐
│ cv_imread()   │  ← OpenCV读取BGR格式
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ BGR→RGB转换   │  ← cv2.cvtColor()
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ PIL Image     │  ← Image.fromarray()
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ CLIP预处理    │  ← self.preprocess()
│  - Resize 224 │
│  - Normalize  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ 模型编码      │  ← encode_image()
│ 输出512维向量 │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ L2归一化      │  ← 关键！/= norm()
└───────┬───────┘
        ▼
   1×512 float32
```

**关键设计决策**:

| 设计点 | 决策 | 理由 |
|:---|:---|:---|
| 图像读取 | OpenCV (BGR) | 更快，兼容性好 |
| 预处理 | CLIP内置 | 与训练时一致 |
| 归一化 | L2归一化 | 使余弦相似度=内积 |
| 文本归一化 | ❌ 不做 | CLIP内部已处理 |
| 设备管理 | 构造函数传入 | 支持CPU/GPU切换 |

**为什么图像要归一化而文本不需要？**

```python
# 图像编码 - 必须归一化
img_feat_vec = self.model.encode_image(image)
img_feat_vec /= img_feat_vec.norm(dim=-1, keepdim=True)  # ✅ 必须！

# 文本编码 - 不需要
feat_text = self.model.encode_text(token)
# feat_text /= feat_text.norm(...)  # ❌ 不需要！
```

原因：CLIP模型的训练方式不同
- 图像编码器输出未归一化，需要手动归一化
- 文本编码器内部已经完成归一化

---

### 1.2 IndexModule - Faiss索引管理

**职责**: 封装Faiss索引的创建、训练和检索

**类定义** (`retrieval_by_faiss.py:37-89`):

```python
class IndexModule:
    def __init__(self, index_string, feat_dim, feat_mat):
        self.index_string = index_string  # "IVF4096,PQ32x8"
        self.feat_dim = feat_dim          # 512
        self.index = None
        self._init_index(feat_mat)

    def _init_index(self, feat_mat):
        # 1. 工厂方法创建索引
        index = faiss.index_factory(self.feat_dim, self.index_string)

        # 2. CPU→GPU迁移 (如果可用)
        if USE_GPU and res is not None:
            self.index = faiss.index_cpu_to_gpu(res, gpu_id, index)

        # 3. 训练索引 (k-means聚类)
        self.index.train(feat_mat)

        # 4. 添加数据
        self.index.add(feat_mat)

    def feat_retrieval(self, feat_query, topk):
        distance, ids = self.index.search(feat_query, topk)
        return distance.squeeze(), ids.squeeze()
```

**索引初始化流程**:

```
特征矩阵 (N×512)
    │
    ▼
┌─────────────────────┐
│ faiss.index_factory │ ← 根据字符串创建索引
│ "IVF4096,PQ32x8"   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  CPU→GPU迁移?       │
│  (如果GPU可用)      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    index.train()    │ ← k-means聚类训练
│   学习4096个中心    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    index.add()      │ ← 添加所有向量
└─────────────────────┘
```

**IVF-PQ参数详解**:

```
IVF4096,PQ32x8
    │      │   │
    │      │   └─ 8bit量化 (每个子码本256个中心)
    │      └───── 分成32个子向量
    └──────────── 4096个倒排列表

内存计算:
- 原始: 120K × 512 × 4B = 240MB
- PQ后: 120K × 32 × 1B = 3.8MB (乘积量化)
- IVF开销: 4096个列表指针
- 总计: ~5MB (压缩比 ~50x)
```

---

### 1.3 ImageRetrievalModule - 检索入口

**职责**: 整合CLIP和Faiss，提供统一检索接口

**类定义** (`retrieval_by_faiss.py:148-176`):

```python
class ImageRetrievalModule:
    def __init__(self, index_string, feat_dim, feat_mat,
                 map_dict, backbone, device):
        self.index_model = IndexModule(index_string, feat_dim, feat_mat)
        self.clip_model = CLIPModel(backbone, device)
        self.map_dict = map_dict  # ID→路径映射

    def retrieval_func(self, query_info, topk):
        # 智能判断输入类型
        if os.path.exists(query_info):      # 图像路径
            feat_vec = self.clip_model.encode_image_by_path(query_info)
        elif isinstance(query_info, str):    # 文本
            feat_vec = self.clip_model.encode_text_by_string(query_info)
        elif isinstance(query_info, np.ndarray):  # 图像数组
            feat_vec = self.clip_model.encode_image_by_ndarray(query_info)

        # 检索
        distance, ids = self.index_model.feat_retrieval(feat_vec, topk)

        # 映射为路径
        paths = [self.map_dict.get(id_tmp, 'None') for id_tmp in ids]
        return distance, ids, paths
```

**输入类型自动检测**:

```python
def retrieval_func(self, query_info, topk):
    # 1. 检查是否为存在的文件路径 (以图搜图)
    if os.path.exists(query_info):
        feat_vec = self.clip_model.encode_image_by_path(query_info)

    # 2. 字符串类型 (以文搜图)
    elif type(query_info) == str:
        feat_vec = self.clip_model.encode_text_by_string(query_info)

    # 3. numpy数组 (从内存图像搜索)
    elif type(query_info) == np.ndarray:
        feat_vec = self.clip_model.encode_image_by_ndarray(query_info)
```

---

## 2. 增量索引层设计

### 2.1 ImageIndexBuilder - 图片特征构建器

**职责**: 封装CLIP特征提取，支持批量处理新图片

**类定义** (`image_index_builder.py:14-96`):

```python
class ImageIndexBuilder:
    def __init__(self, device=None):
        self.device = device or CFG.device
        self.model, self.preprocess = clip.load(
            CFG.clip_backbone_type,
            device=self.device,
            jit=False
        )

    def extract_features(self, image_paths: List[str]) -> Tuple[np.ndarray, List[str]]
    def build_mapping(self, image_paths: List[str], start_id: int) -> Dict[int, str]
```

**特征提取流程**:

```
图片路径列表
    │
    ▼
┌─────────────────────┐
│ 遍历每张图片        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 读取图片 (OpenCV)   │
│ - 错误处理: 跳过    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 颜色空间转换        │
│ BGR → RGB           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ CLIP预处理          │
│ - Resize 224        │
│ - Normalize         │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 模型推理            │
│ encode_image()      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ L2归一化            │
│ /= norm()           │
└─────────┬───────────┘
          │
          ▼
   特征矩阵 (N×512)
```

**关键设计**:

| 设计点 | 说明 |
|:---|:---|
| **错误处理** | 单张图片失败不影响整体，返回有效路径列表 |
| **批量处理** | 支持多图片批量提取，带tqdm进度条 |
| **特征归一化** | 自动L2归一化，确保可比性 |
| **ID分配** | 支持从指定start_id开始分配 |

---

### 2.2 IncrementalIndexManager - 增量索引管理器

**职责**: 管理主索引和缓冲区，协调增删改查操作

**类定义** (`incremental_index_manager.py:17-393`):

```python
class IncrementalIndexManager:
    def __init__(self, main_index, index_builder,
                 buffer_size_threshold=1000, state_dir="./data/index_state"):
        self.main_index = main_index          # Faiss主索引
        self.index_builder = index_builder    # 特征提取器
        self.buffer_size_threshold = buffer_size_threshold

        # 缓冲区存储
        self.buffer_features = []   # 特征列表
        self.buffer_paths = []      # 路径列表
        self.buffer_ids = []        # ID列表

        # 删除标记
        self.deleted_ids = set()

        # ID分配计数器
        self.next_id = main_index.get_total_count()
```

#### 2.2.1 添加图片流程

```
用户请求: add_images([path1, path2])
    │
    ▼
┌─────────────────────┐
│ 特征提取            │
│ ImageIndexBuilder   │
│ .extract_features() │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 分配新ID            │
│ range(next_id, ...) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 存入缓冲区          │
│ - buffer_features   │
│ - buffer_paths      │
│ - buffer_ids        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 更新映射字典        │
│ id_to_path[id] = path
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 检查阈值?           │
└─────────┬───────────┘
     是/超过阈值
          │
          ▼
┌─────────────────────┐
│ 提示合并            │
│ "建议执行merge"     │
└─────────────────────┘
```

#### 2.2.2 删除图片流程（逻辑删除）

```python
def remove_images(self, image_ids: List[int]) -> Dict:
    """
    逻辑删除流程
    1. 验证ID存在性（主索引或缓冲区）
    2. 添加到deleted_ids集合
    3. 如果在缓冲区，立即移除
    4. 返回删除结果
    """
    for id_ in image_ids:
        # 检查存在性
        exists_in_main = id_ < self.main_index.get_total_count()
        exists_in_buffer = id_ in self.buffer_ids

        if not exists_in_main and not exists_in_buffer:
            continue  # ID不存在

        # 逻辑删除：添加到删除集合
        self.deleted_ids.add(id_)

        # 缓冲区可立即移除（优化）
        if id_ in self.buffer_ids:
            idx = self.buffer_ids.index(id_)
            self.buffer_features.pop(idx)
            self.buffer_paths.pop(idx)
            self.buffer_ids.pop(idx)
```

**为什么用逻辑删除？**

| 方案 | 优点 | 缺点 |
|:---|:---|:---|
| **物理删除** | 立即释放空间 | Faiss IVF不支持，必须重建索引（2-4小时） |
| **逻辑删除** | 即时生效，无需重建 | 占用少量内存，需定期重建清理 |

#### 2.2.3 搜索流程（双路合并）

```python
def search(self, query_vector, topk=5):
    """
    搜索流程：合并主索引和缓冲区结果
    """
    # 1. 搜索主索引
    main_distances, main_ids = self._search_main_index(query_vector, topk*2)

    # 2. 搜索缓冲区（暴力搜索）
    buffer_distances, buffer_ids, buffer_paths = \
        self._search_buffer(query_vector, topk*2)

    # 3. 合并结果
    all_results = []

    # 添加主索引结果（过滤已删除）
    for dist, id_ in zip(main_distances, main_ids):
        if id_ not in self.deleted_ids:
            all_results.append((dist, id_, self._get_path_by_id(id_)))

    # 添加缓冲区结果（过滤已删除）
    for dist, id_, path in zip(buffer_distances, buffer_ids, buffer_paths):
        if id_ not in self.deleted_ids:
            all_results.append((dist, id_, path))

    # 4. 按距离排序，取topk
    all_results.sort(key=lambda x: x[0])
    return all_results[:topk]
```

**缓冲区暴力搜索原理**:

```python
def _search_buffer(self, query_vector, topk):
    """
    缓冲区使用暴力搜索（数据量小）
    """
    # 合并缓冲区特征
    buffer_matrix = np.vstack(self.buffer_features)  # (N, 512)

    # 计算余弦相似度（向量已归一化，点积=余弦相似度）
    similarities = np.dot(buffer_matrix, query_vector.T).flatten()

    # 转换为距离
    distances = 1 - similarities

    # 取topk
    top_indices = np.argsort(distances)[:topk]

    return distances[top_indices], \
           [self.buffer_ids[i] for i in top_indices], \
           [self.buffer_paths[i] for i in top_indices]
```

#### 2.2.4 状态持久化设计

```python
def save_state(self):
    """
    保存状态到磁盘
    """
    # 1. 保存状态JSON（可读）
    state = {
        "next_id": self.next_id,
        "deleted_ids": list(self.deleted_ids),
        "buffer_ids": self.buffer_ids,
        "buffer_paths": self.buffer_paths,
    }
    with open(f"{state_dir}/index_state.json", 'w') as f:
        json.dump(state, f)

    # 2. 保存缓冲区特征（Pickle，高效）
    if len(self.buffer_features) > 0:
        buffer_matrix = np.vstack(self.buffer_features)
        with open(f"{state_dir}/buffer_features.pkl", 'wb') as f:
            pickle.dump(buffer_matrix, f)

def load_state(self, map_dict):
    """
    从磁盘加载状态
    """
    # 加载JSON状态
    with open(f"{state_dir}/index_state.json", 'r') as f:
        state = json.load(f)

    self.next_id = state["next_id"]
    self.deleted_ids = set(state["deleted_ids"])
    self.buffer_ids = state["buffer_ids"]
    self.buffer_paths = state["buffer_paths"]

    # 加载Pickle特征
    with open(f"{state_dir}/buffer_features.pkl", 'rb') as f:
        buffer_matrix = pickle.load(f)

    # 还原为列表
    self.buffer_features = [
        buffer_matrix[i:i+1] for i in range(len(self.buffer_ids))
    ]
```

---

## 3. RAG引擎层设计

### 3.1 RAGEngine - 核心编排器

**职责**: 整合检索与LLM能力，实现智能检索流程，支持增量索引

**核心方法** (`rag_engine.py:21-275`):

```python
class RAGEngine:
    def __init__(self, retrieval_module, llm_interface,
                 enable_expansion=True, enable_explanation=True,
                 index_manager=None):  # 新增：增量索引管理器
        self.retrieval_module = retrieval_module
        self.llm = llm_interface
        self.index_manager = index_manager  # 增量索引支持
        self.enable_expansion = enable_expansion and self.llm.available
        self.enable_explanation = enable_explanation and self.llm.available
```

#### 3.1.1 增量索引搜索集成

```python
def _search(self, query: str, topk: int) -> tuple:
    """
    内部搜索方法，支持增量索引管理器
    """
    if self.index_manager is not None:
        # 使用增量索引管理器搜索
        # 1. 将查询转换为特征向量
        if os.path.exists(query):
            feat_vec = self.retrieval_module.clip_model.encode_image_by_path(query)
        else:
            feat_vec = self.retrieval_module.clip_model.encode_text_by_string(query)

        feat_vec = feat_vec.astype(np.float32)

        # 2. 调用增量索引搜索（包含主索引+缓冲区，过滤已删除）
        distances, ids, paths = self.index_manager.search(feat_vec, topk)
        return distances, ids, paths
    else:
        # 回退到传统检索模块
        return self.retrieval_module.retrieval_func(query, topk)
```

#### 3.1.2 智能检索流程 (search_and_explain)

```
用户查询
    │
    ▼
┌─────────────────────┐
│  查询扩展?          │
│  (文本且LLM可用)    │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
  是           否
    │           │
    ▼           │
┌──────────┐    │
│ LLM扩展  │    │
│ 生成N个  │    │
│ 变体查询 │    │
└────┬─────┘    │
     │          │
     └────┬─────┘
          ▼
┌─────────────────────┐
│  多查询并行检索     │
│  (每个查询检索     │
│   topk*2个结果)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  结果融合           │
│  - Set去重          │
│  - 保留最小距离     │
│  - 按距离排序       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  截断取TopK         │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  AI解释?            │
│  (LLM可用)          │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
  是           否
    │           │
    ▼           │
┌──────────┐    │
│ LLM生成  │    │
│ 结果解释 │    │
└────┬─────┘    │
     │          │
     └────┬─────┘
          ▼
   返回结果字典
```

**核心代码** (`rag_engine.py:83-129`):

```python
def search_and_explain(self, query, topk=10, use_expansion=True):
    result = {
        "original_query": query,
        "expanded_queries": [],
        "results": [],
        "ai_explanation": "",
        "total_results": 0
    }

    # Step 1: 查询扩展
    expanded_queries = []
    if use_expansion and self.enable_expansion and not os.path.exists(query):
        expanded_queries = self.llm.expand_query(query, num_expansions=3)
        result["expanded_queries"] = expanded_queries
    else:
        expanded_queries = [query]

    # Step 2: 多查询检索与融合
    all_results = []
    seen_paths = set()

    for q in expanded_queries:
        distance_result, index_result, path_list = \
            self.retrieval_module.retrieval_func(q, topk=min(topk * 2, 20))

        for dist, idx, path in zip(distance_result, index_result, path_list):
            if path not in seen_paths and path != 'None':
                seen_paths.add(path)
                all_results.append({
                    "path": path,
                    "distance": float(dist),
                    "index": int(idx),
                    "matched_query": q
                })

    # Step 3: 重排序（按距离排序）
    all_results.sort(key=lambda x: x["distance"])
    final_results = all_results[:topk]

    # Step 4: AI解释
    if self.enable_explanation and final_results:
        explanation = self.llm.explain_results(query, final_results)
        result["ai_explanation"] = explanation

    result["results"] = final_results
    result["total_results"] = len(final_results)
    return result
```

#### 3.1.3 RAG问答 (rag_qa)

```python
def rag_qa(self, question: str, topk: int = 5) -> Dict[str, Any]:
    # Step 1: 检索相关图片
    retrieval_result = self.search_and_explain(
        query=question,
        topk=topk,
        use_expansion=False  # QA场景通常不需要扩展
    )

    # Step 2: LLM基于上下文回答
    if self.llm.available:
        answer = self.llm.answer_with_rag(
            query=question,
            context=retrieval_result["results"]
        )
    else:
        answer = "（LLM未配置，请配置LLM_API_KEY以启用智能回答）"

    return {
        "question": question,
        "answer": answer,
        "context": retrieval_result["results"],
        "ai_explanation": retrieval_result.get("ai_explanation", "")
    }
```

---

### 3.2 LLMInterface - LLM能力封装

**职责**: 封装所有LLM调用，提供查询扩展、结果解释、RAG问答等功能

**设计模式**: 策略模式 + 工厂模式

**类结构**:

```python
class LLMInterface:
    def __init__(self, base_url=None, api_key=None,
                 model="gpt-3.5-turbo", temperature=0.7):
        # 优先从环境变量读取
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")

        # 检查可用性
        self._available = bool(self.api_key)
        if self._available:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @property
    def available(self) -> bool:
        """Graceful degradation 关键"""
        return self._available

    def expand_query(self, user_query: str, num_expansions: int = 3) -> List[str]
    def explain_results(self, query: str, results: List[Dict]) -> str
    def answer_with_rag(self, query: str, context: List[Dict]) -> str
    def chat_with_context(self, query: str, context: List[Dict],
                          history: Optional[List[Dict]] = None) -> str
```

#### 3.2.1 查询扩展实现

**Prompt设计** (`llm_interface.py:74-84`):

```python
prompt = f"""作为图像搜索助手，请基于用户的查询生成{num_expansions}个不同表达的搜索意图，以提高检索召回率。

用户查询: {user_query}

要求:
1. 保持原始语义，但使用不同的词汇和表达方式
2. 可以从不同角度描述（如风格、场景、对象、颜色等）
3. 每个扩展查询简洁明了，不超过20个字
4. 直接返回扩展查询列表，每行一个，不要编号

输出:"""
```

**示例**:

```
输入: "a dog playing in the park"

输出:
a dog playing in the park
dog running on grass
puppy playing outdoors
pet in the garden
canine enjoying nature
```

#### 3.2.2 结果解释实现

**Prompt设计** (`llm_interface.py:134-146`):

```python
prompt = f"""作为图像检索系统的AI助手，请为用户解释以下检索结果的相关性。

用户查询: {query}

检索结果（按相似度排序）:
{results_text}

请用简洁友好的语言（100字以内）:
1. 简要说明检索结果与用户查询的匹配程度
2. 指出最相关的结果及其特点
3. 如果有异常情况（如结果不太相关），给出可能的原因

输出:"""
```

---

## 3. Agent工具层设计

### 3.1 AgentTool - 工具定义

**职责**: 标准化工具定义，支持OpenAI Function Calling格式

**数据结构** (`agent_tools.py:22-86`):

```python
@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[str]] = None

@dataclass
class AgentTool:
    name: str
    description: str
    parameters: List[ToolParameter]
    func: Callable

    def to_openai_function(self) -> Dict[str, Any]:
        """转换为OpenAI Function Calling格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
```

### 3.2 ImageRetrievalTools - 工具集合

**职责**: 管理所有图像检索相关工具

**内置工具列表**:

| 工具名 | 功能 | 参数 |
|:---|:---|:---|
| `search_by_text` | 文本搜索图片 | query, topk |
| `search_by_image` | 以图搜图 | image_path, topk |
| `explain_search_results` | 解释搜索结果 | original_query, results |
| `answer_with_image_context` | 基于图片问答 | question, context_size |

**工具注册示例** (`agent_tools.py:113-133`):

```python
self.register_tool(
    AgentTool(
        name="search_by_text",
        description="根据文本描述搜索相关图片。支持自然语言描述...",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="搜索文本描述",
                required=True
            ),
            ToolParameter(
                name="topk",
                type="integer",
                description="返回结果数量，默认为5",
                required=False
            )
        ],
        func=self._search_by_text
    )
)
```

**工具执行流程**:

```
API请求 /api/tools/call
    │
    ├─→ tool_name: "search_by_text"
    ├─→ arguments: {"query": "a dog", "topk": 5}
    │
    ▼
┌─────────────────────┐
│ ImageRetrievalTools │
│ .execute_tool()     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 查找工具 (get_tool) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 执行工具函数        │
│ func(**arguments)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 包装结果            │
│ {"tool", "result",   │
│  "success"}         │
└─────────────────────┘
```

### 3.3 ToolUsingAgent - 工具使用演示

**职责**: 演示如何让LLM使用工具进行多步推理

**核心流程** (`agent_tools.py:344-416`):

```python
class ToolUsingAgent:
    def run(self, user_input: str) -> Dict[str, Any]:
        # 1. 构建系统提示（包含工具描述）
        system_prompt = f"""你是一个智能图像检索助手。你可以使用以下工具:
        {self.tools.get_tools_description()}

        直接以JSON格式返回工具调用:
        {{"tool": "tool_name", "arguments": {{"arg1": "value1"}}}}"""

        # 2. 调用LLM决定使用什么工具
        response = self.llm.client.chat.completions.create(...)

        # 3. 解析工具调用
        tool_call = json.loads(response.content)
        tool_name = tool_call.get("tool")
        arguments = tool_call.get("arguments")

        # 4. 执行工具
        result = self.tools.execute_tool(tool_name, arguments)

        return {
            "user_input": user_input,
            "tool_call": tool_call,
            "result": result
        }
```

---

## 4. 应用层设计

### 4.1 Flask应用架构

**职责**: HTTP请求处理、API路由、结果封装

**初始化流程** (`flask_app.py:19-46`):

```python
# 1. 加载特征数据
with open(CFG.feat_mat_path, 'rb') as f:
    feat_mat = pickle.load(f)
with open(CFG.map_dict_path, 'rb') as f:
    map_dict = pickle.load(f)

# 2. 初始化检索模块
ir_model = ImageRetrievalModule(
    CFG.index_string, CFG.feat_dim, feat_mat, map_dict,
    CFG.clip_backbone_type, CFG.device
)

# 3. 初始化RAG引擎
llm_interface = LLMInterface(...)
rag_engine = RAGEngine(
    retrieval_module=ir_model,
    llm_interface=llm_interface,
    enable_expansion=CFG.rag_enable_expansion,
    enable_explanation=CFG.rag_enable_explanation
)

# 4. 初始化Agent工具（条件式）
if CFG.agent_tools_enabled and llm_interface.available:
    agent_tools = create_tools_for_rag_engine(rag_engine)
```

### 4.2 API端点设计

#### RAG检索端点 (`flask_app.py:92-121`)

```python
@app.route('/api/search/rag', methods=['POST'])
def search_rag():
    data = request.get_json() or {}
    query = data.get('query', '')
    topk = data.get('topk', CFG.topk)
    use_expansion = data.get('use_expansion', CFG.rag_enable_expansion)

    result = rag_engine.search_and_explain(
        query=query,
        topk=topk,
        use_expansion=use_expansion
    )

    # 转换路径为URL
    for r in result.get('results', []):
        r['url'] = 'static/img/' + os.path.basename(r['path'])

    return jsonify(result)
```

#### Agent工具调用端点 (`flask_app.py:231-261`)

```python
@app.route('/api/tools/call', methods=['POST'])
def call_tool():
    if not agent_tools:
        return jsonify({"error": "Agent tools not available"}), 503

    data = request.get_json() or {}
    tool_name = data.get('tool', '')
    arguments = data.get('arguments', {})

    result = agent_tools.execute_tool(tool_name, arguments)

    # 转换结果路径
    if result.get('success'):
        tool_result = result.get('result', {})
        if 'results' in tool_result:
            for r in tool_result['results']:
                if 'path' in r:
                    r['url'] = 'static/img/' + os.path.basename(r['path'])

    return jsonify(result)
```

---

## 5. 配置管理设计

### 5.1 配置分层

```python
# config/base_config.py

# 第1层：代码默认配置
CFG.llm_model = 'gpt-3.5-turbo'
CFG.rag_enable_expansion = True

# 第2层：.env文件 (通过python-dotenv加载)
load_dotenv('.env')

# 第3层：环境变量 (运行时覆盖)
CFG.llm_api_key = os.getenv('LLM_API_KEY', '')
```

### 5.2 配置项分类

| 类别 | 配置项 | 说明 |
|:---|:---|:---|
| **模型配置** | clip_backbone_type | CLIP模型类型 |
| | device | cuda/cpu |
| **数据配置** | image_file_dir | 图像数据集路径 |
| | database_dir | 特征存储路径 |
| **索引配置** | index_string | Faiss索引类型 |
| | feat_dim | 特征维度 |
| | topk | 默认返回数量 |
| **LLM配置** | llm_base_url | API地址 |
| | llm_api_key | API密钥 |
| | llm_model | 模型名称 |
| **RAG配置** | rag_enable_expansion | 查询扩展开关 |
| | rag_enable_explanation | AI解释开关 |
| | rag_context_size | 上下文数量 |
| **Agent配置** | agent_tools_enabled | 工具开关 |

---

## 6. 错误处理设计

### 6.1 分层错误处理策略

| 层级 | 错误类型 | 处理策略 |
|:---|:---|:---|
| **应用层** | 参数错误 | 返回400状态码 + 错误信息 |
| | LLM不可用 | 返回503状态码 |
| **RAG引擎层** | 单查询失败 | 记录日志，继续其他查询 |
| | LLM调用失败 | 返回降级结果 |
| **检索层** | 文件不存在 | 返回'None'占位 |
| | 索引未找到 | 返回-1索引 |

### 6.2 关键错误处理代码

```python
# rag_engine.py:97-114 单查询失败不影响整体
for q in expanded_queries:
    try:
        distance_result, index_result, path_list = \
            self.retrieval_module.retrieval_func(q, topk=min(topk * 2, 20))
        # ... 处理结果
    except Exception as e:
        print(f"[RAGEngine] 检索失败 '{q}': {e}")
        continue  # 跳过失败的查询，继续下一个

# llm_interface.py:71-72, 108-110 LLM失败返回默认值
def expand_query(self, user_query: str, num_expansions: int = 3):
    if not self._available:
        return [user_query]  # 降级：返回原始查询

    try:
        # ... LLM调用
    except Exception as e:
        print(f"[LLMInterface] 查询扩展失败: {e}")
        return [user_query]  # 失败时返回原始查询
```

---

## 7. 扩展设计模式

### 7.1 添加新工具的模板

```python
def _register_custom_tools(self):
    """添加自定义工具示例"""

    self.register_tool(
        AgentTool(
            name="filter_by_color",
            description="按颜色过滤检索结果",
            parameters=[
                ToolParameter(
                    name="color",
                    type="string",
                    description="目标颜色",
                    required=True,
                    enum=["red", "blue", "green", "yellow"]
                )
            ],
            func=self._filter_by_color
        )
    )

def _filter_by_color(self, color: str) -> Dict[str, Any]:
    """工具实现"""
    # 实现过滤逻辑
    return {"filtered_results": [...]}
```

### 7.2 RAG Pipeline扩展

```python
# 创建自定义Pipeline
def custom_rag_pipeline():
    pipeline = RAGPipeline(retrieval_module, llm)

    # 添加自定义步骤
    pipeline.add_step("retrieve", retrieve_step)
    pipeline.add_step("rerank", cross_encoder_rerank_step)  # 重排序
    pipeline.add_step("filter", diversity_filter_step)       # 多样性过滤
    pipeline.add_step("explain", explain_step)

    return pipeline
```

---

**上一章**: [01 - 系统架构总览](./01-architecture-overview.md)

**下一章**: [03 - 关键算法实现](./03-key-algorithms.md) - 深入算法细节和代码实现
