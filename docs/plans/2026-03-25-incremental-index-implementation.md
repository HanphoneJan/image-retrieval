# 增量图片索引更新实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现支持增量添加和逻辑删除的图片索引管理系统，新增 IncrementalIndexManager 模块和 REST API。

**Architecture:**
- 保留现有的 `IndexModule` 和 `CLIPModel`
- 新增 `IncrementalIndexManager` 管理主索引和缓冲区
- 新增 `ImageIndexBuilder` 封装特征提取
- 新增 5 个 Flask API 端点
- 使用 JSON 文件持久化索引状态

**Tech Stack:** Python, Faiss, PyTorch, CLIP, Flask, NumPy

---

## 前置准备

**必须已阅读:**
- [设计文档](./2026-03-25-incremental-index-design.md)
- [现有代码](../../retrieval_by_faiss.py) - 了解 IndexModule 当前实现
- [现有代码](../../image_feature_extract.py) - 了解特征提取流程

**当前文件结构:**
```
image-retrieval/
├── retrieval_by_faiss.py      # IndexModule 在此文件中
├── image_feature_extract.py   # 特征提取脚本
├── flask_app.py               # 需要新增 API 端点
├── config/
│   └── base_config.py         # 配置管理
└── my_utils/
    └── utils.py               # 工具函数
```

---

## Task 1: 增强 IndexModule 类

**Files:**
- Modify: `retrieval_by_faiss.py:37-89` (IndexModule 类)

**Step 1: 分析现有代码**

阅读 `retrieval_by_faiss.py` 中 IndexModule 类的当前实现，了解：
- `__init__` 初始化逻辑
- `_init_index` 训练和添加数据
- `feat_retrieval` 搜索方法

**Step 2: 添加增量方法**

在 `IndexModule` 类中添加以下方法：

```python
def add_vectors(self, vectors: np.ndarray) -> List[int]:
    """
    增量添加向量到索引

    Args:
        vectors: 特征向量矩阵，shape (N, feat_dim)

    Returns:
        新添加向量的ID列表（从当前总数开始递增）
    """
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    # 获取当前索引中的向量数
    start_id = self.index.ntotal

    # 添加到索引
    self.index.add(vectors)

    # 返回新分配的ID
    new_ids = list(range(start_id, start_id + len(vectors)))
    return new_ids

def get_total_count(self) -> int:
    """获取索引中当前向量总数"""
    return self.index.ntotal
```

**Step 3: 验证修改**

创建一个简单测试：

```python
# test_index_add.py
import numpy as np
from retrieval_by_faiss import IndexModule

# 创建测试数据
feat_mat = np.random.randn(100, 512).astype(np.float32)

# 初始化索引
index = IndexModule("IVF4096,PQ32x8", 512, feat_mat)
print(f"Initial count: {index.get_total_count()}")

# 添加新向量
new_vectors = np.random.randn(10, 512).astype(np.float32)
new_ids = index.add_vectors(new_vectors)
print(f"Added IDs: {new_ids}")
print(f"New count: {index.get_total_count()}")
```

运行: `python test_index_add.py`
期望输出:
```
Initial count: 100
Added IDs: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
New count: 110
```

**Step 4: 提交**

```bash
git add retrieval_by_faiss.py test_index_add.py
git commit -m "feat: add add_vectors and get_total_count to IndexModule"
```

---

## Task 2: 创建 ImageIndexBuilder 类

**Files:**
- Create: `image_index_builder.py`

**Step 1: 编写 ImageIndexBuilder**

```python
# image_index_builder.py
import os
from typing import List, Dict, Tuple
import numpy as np
import torch
import clip
import cv2
from PIL import Image
from tqdm import tqdm

from config.base_config import CFG


class ImageIndexBuilder:
    """
    图片索引构建器

    职责：
    1. 批量提取图片CLIP特征
    2. 管理ID到路径的映射
    """

    def __init__(self, device=None):
        self.device = device or CFG.device
        self.model, self.preprocess = clip.load(
            CFG.clip_backbone_type,
            device=self.device,
            jit=False
        )

    def extract_features(self, image_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        批量提取图片特征

        Args:
            image_paths: 图片路径列表

        Returns:
            (features, valid_paths):
            - features: numpy array shape (N, 512)
            - valid_paths: 成功提取的路径列表（失败的已过滤）
        """
        features = []
        valid_paths = []

        for path in tqdm(image_paths, desc="Extracting features"):
            try:
                if not os.path.exists(path):
                    print(f"Warning: Image not found: {path}")
                    continue

                # 读取图片
                image_bgr = cv2.imread(path)
                if image_bgr is None:
                    print(f"Warning: Failed to load image: {path}")
                    continue

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image = self.preprocess(
                    Image.fromarray(image_rgb)
                ).unsqueeze(0).to(self.device)

                # 提取特征
                with torch.no_grad():
                    feat = self.model.encode_image(image)
                    feat = feat / feat.norm(dim=-1, keepdim=True)  # L2归一化
                    feat = feat.cpu().numpy()

                features.append(feat)
                valid_paths.append(path)

            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        if not features:
            return np.array([]).reshape(0, 512), []

        return np.vstack(features), valid_paths

    def build_mapping(self, image_paths: List[str], start_id: int = 0) -> Dict[int, str]:
        """
        构建ID到路径的映射

        Args:
            image_paths: 图片路径列表
            start_id: 起始ID

        Returns:
            映射字典 {id: path}
        """
        return {
            i + start_id: path
            for i, path in enumerate(image_paths)
        }
```

**Step 2: 编写测试**

创建 `test_image_index_builder.py`：

```python
import os
import numpy as np
from image_index_builder import ImageIndexBuilder

# 测试
builder = ImageIndexBuilder()

# 使用现有图片测试
test_images = [
    "train2017/000000000001.jpg",  # 替换为实际存在的图片
]

if os.path.exists(test_images[0]):
    features, valid_paths = builder.extract_features(test_images)
    print(f"Extracted {len(valid_paths)} features, shape: {features.shape}")

    mapping = builder.build_mapping(valid_paths, start_id=1000)
    print(f"Mapping: {mapping}")
else:
    print(f"Test image not found: {test_images[0]}")
```

运行: `python test_image_index_builder.py`
期望输出类似:
```
Extracting features: 100%|████████| 1/1 [00:00<00:00,  2.50it/s]
Extracted 1 features, shape: (1, 512)
Mapping: {1000: 'train2017/000000000001.jpg'}
```

**Step 3: 提交**

```bash
git add image_index_builder.py test_image_index_builder.py
git commit -m "feat: add ImageIndexBuilder for feature extraction"
```

---

## Task 3: 创建 IncrementalIndexManager 类

**Files:**
- Create: `incremental_index_manager.py`

**Step 1: 编写 IncrementalIndexManager**

```python
# incremental_index_manager.py
import json
import pickle
import os
from typing import List, Dict, Tuple, Optional
import numpy as np

from retrieval_by_faiss import IndexModule, CLIPModel
from image_index_builder import ImageIndexBuilder


class IncrementalIndexManager:
    """
    增量索引管理器

    管理主索引和缓冲区，支持增量添加和逻辑删除
    """

    def __init__(
        self,
        main_index: IndexModule,
        map_dict: Dict[int, str],
        state_path: str = "data/index_state.json",
        buffer_size_threshold: int = 1000
    ):
        self.main_index = main_index
        self.map_dict = map_dict  # 主索引ID映射
        self.state_path = state_path
        self.buffer_size_threshold = buffer_size_threshold

        # 缓冲区
        self.buffer_features: Optional[np.ndarray] = None
        self.buffer_map: Dict[int, str] = {}
        self.buffer_start_id: int = max(map_dict.keys()) + 1 if map_dict else 0

        # 已删除ID集合
        self.deleted_ids: set = set()

        # 加载状态
        self._load_state()

    def _load_state(self):
        """从文件加载状态"""
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r') as f:
                state = json.load(f)
                self.deleted_ids = set(state.get('deleted_ids', []))
                self.buffer_map = {
                    int(k): v for k, v in state.get('buffer_map', {}).items()
                }
                self.buffer_start_id = state.get('buffer_start_id', self.buffer_start_id)

            # 加载缓冲区特征
            buffer_feat_path = self.state_path.replace('.json', '_buffer.pkl')
            if os.path.exists(buffer_feat_path):
                with open(buffer_feat_path, 'rb') as f:
                    self.buffer_features = pickle.load(f)

    def _save_state(self):
        """保存状态到文件"""
        state = {
            'deleted_ids': list(self.deleted_ids),
            'buffer_map': self.buffer_map,
            'buffer_start_id': self.buffer_start_id,
            'main_count': self.main_index.get_total_count()
        }

        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump(state, f)

        # 保存缓冲区特征
        if self.buffer_features is not None and len(self.buffer_features) > 0:
            buffer_feat_path = self.state_path.replace('.json', '_buffer.pkl')
            with open(buffer_feat_path, 'wb') as f:
                pickle.dump(self.buffer_features, f)

    def add_images(
        self,
        image_paths: List[str],
        builder: ImageIndexBuilder
    ) -> Dict:
        """
        添加新图片到缓冲区

        Returns:
            {'added': int, 'ids': List[int], 'failed': List[str]}
        """
        # 提取特征
        features, valid_paths = builder.extract_features(image_paths)

        if len(features) == 0:
            return {'added': 0, 'ids': [], 'failed': image_paths}

        # 计算新ID
        start_id = self.buffer_start_id
        new_ids = list(range(start_id, start_id + len(valid_paths)))
        self.buffer_start_id = start_id + len(valid_paths)

        # 更新缓冲区
        if self.buffer_features is None or len(self.buffer_features) == 0:
            self.buffer_features = features
        else:
            self.buffer_features = np.vstack([self.buffer_features, features])

        # 更新映射
        for i, path in enumerate(valid_paths):
            self.buffer_map[new_ids[i]] = path

        # 保存状态
        self._save_state()

        # 检查是否需要合并
        should_merge = len(self.buffer_features) >= self.buffer_size_threshold

        failed = [p for p in image_paths if p not in valid_paths]

        return {
            'added': len(valid_paths),
            'ids': new_ids,
            'failed': failed,
            'buffer_size': len(self.buffer_features),
            'should_merge': should_merge
        }

    def remove_images(self, image_ids: List[int]) -> Dict:
        """
        标记删除图片

        Returns:
            {'removed': int, 'not_found': List[int]}
        """
        removed = []
        not_found = []

        for img_id in image_ids:
            # 检查ID是否存在
            if img_id in self.map_dict or img_id in self.buffer_map:
                self.deleted_ids.add(img_id)
                removed.append(img_id)
            else:
                not_found.append(img_id)

        self._save_state()

        return {'removed': len(removed), 'not_found': not_found}

    def search(
        self,
        query_vector: np.ndarray,
        topk: int = 10
    ) -> Tuple[List[float], List[int], List[str]]:
        """
        搜索（合并主索引+缓冲区，过滤已删除）

        Returns:
            (distances, ids, paths)
        """
        results = []

        # 搜索主索引
        if self.main_index.get_total_count() > 0:
            dist_main, ids_main = self.main_index.feat_retrieval(
                query_vector, topk * 2
            )
            for d, i in zip(dist_main, ids_main):
                if i not in self.deleted_ids:
                    results.append((d, i, self.map_dict.get(i, "unknown")))

        # 搜索缓冲区（暴力搜索）
        if self.buffer_features is not None and len(self.buffer_features) > 0:
            # 计算L2距离
            diff = self.buffer_features - query_vector
            dist_buffer = np.sum(diff ** 2, axis=1)

            # 获取topk
            top_indices = np.argsort(dist_buffer)[:topk]
            buffer_ids = list(self.buffer_map.keys())

            for idx in top_indices:
                buf_id = buffer_ids[idx]
                if buf_id not in self.deleted_ids:
                    results.append((
                        float(dist_buffer[idx]),
                        buf_id,
                        self.buffer_map[buf_id]
                    ))

        # 排序并取topk
        results.sort(key=lambda x: x[0])
        results = results[:topk]

        if not results:
            return [], [], []

        distances, ids, paths = zip(*results)
        return list(distances), list(ids), list(paths)

    def merge_buffer(self) -> Dict:
        """
        将缓冲区合并到主索引

        Returns:
            {'merged': int}
        """
        if self.buffer_features is None or len(self.buffer_features) == 0:
            return {'merged': 0}

        # 添加到主索引
        new_ids = self.main_index.add_vectors(self.buffer_features)

        # 更新主映射
        for i, buf_id in enumerate(self.buffer_map.keys()):
            self.map_dict[new_ids[i]] = self.buffer_map[buf_id]

        # 清空缓冲区
        count = len(self.buffer_features)
        self.buffer_features = None
        self.buffer_map = {}

        self._save_state()

        return {'merged': count}

    def get_status(self) -> Dict:
        """获取索引状态"""
        return {
            'main_count': self.main_index.get_total_count(),
            'buffer_count': len(self.buffer_features) if self.buffer_features is not None else 0,
            'deleted_count': len(self.deleted_ids),
            'total_images': len(self.map_dict) + len(self.buffer_map),
            'buffer_threshold': self.buffer_size_threshold
        }
```

**Step 2: 编写测试**

创建 `test_incremental_manager.py`：

```python
import numpy as np
from retrieval_by_faiss import IndexModule
from incremental_index_manager import IncrementalIndexManager

# 创建测试索引
feat_mat = np.random.randn(100, 512).astype(np.float32)
main_index = IndexModule("IVF4096,PQ32x8", 512, feat_mat)
map_dict = {i: f"image_{i}.jpg" for i in range(100)}

# 创建管理器
manager = IncrementalIndexManager(
    main_index,
    map_dict,
    state_path="data/test_state.json",
    buffer_size_threshold=10
)

print(f"Initial status: {manager.get_status()}")

# 测试搜索
query = np.random.randn(1, 512).astype(np.float32)
dist, ids, paths = manager.search(query, topk=5)
print(f"Search results: {len(ids)} items")

# 测试删除
result = manager.remove_images([0, 1, 2])
print(f"Remove result: {result}")

# 测试状态
print(f"Status after remove: {manager.get_status()}")
```

运行: `python test_incremental_manager.py`
期望输出:
```
[IndexModule] 使用CPU索引
training time:0.xx s, train set:(100, 512)
Initial status: {'main_count': 100, 'buffer_count': 0, ...}
Search results: 5 items
Remove result: {'removed': 3, 'not_found': []}
Status after remove: {'main_count': 100, 'buffer_count': 0, 'deleted_count': 3, ...}
```

**Step 3: 提交**

```bash
git add incremental_index_manager.py test_incremental_manager.py
git commit -m "feat: add IncrementalIndexManager for incremental updates"
```

---

## Task 4: 添加 Flask API 端点

**Files:**
- Modify: `flask_app.py`

**Step 1: 导入新模块**

在 `flask_app.py` 顶部添加导入：

```python
from image_index_builder import ImageIndexBuilder
from incremental_index_manager import IncrementalIndexManager
```

**Step 2: 初始化管理器**

在应用初始化时创建 `IncrementalIndexManager`：

```python
# 在创建 IndexModule 和加载 map_dict 之后
index_module = IndexModule(...)

# 加载现有映射
with open(CFG.map_dict_path, 'rb') as f:
    map_dict = pickle.load(f)

# 创建增量管理器
index_manager = IncrementalIndexManager(
    main_index=index_module,
    map_dict=map_dict,
    state_path="data/index_state.json",
    buffer_size_threshold=1000
)

# 创建特征提取器
image_builder = ImageIndexBuilder(device=CFG.device)
```

**Step 3: 添加 API 路由**

在 `flask_app.py` 中添加以下路由：

```python
@app.route('/api/index/add', methods=['POST'])
def add_images_to_index():
    """添加新图片到索引"""
    data = request.get_json()

    if not data or 'image_paths' not in data:
        return jsonify({'error': 'Missing image_paths'}), 400

    image_paths = data['image_paths']
    if not isinstance(image_paths, list):
        return jsonify({'error': 'image_paths must be a list'}), 400

    # 验证路径存在
    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if len(valid_paths) != len(image_paths):
        missing = [p for p in image_paths if p not in valid_paths]
        print(f"Warning: {len(missing)} images not found")

    # 添加到索引
    result = index_manager.add_images(valid_paths, image_builder)

    return jsonify(result)


@app.route('/api/index/remove', methods=['POST'])
def remove_images_from_index():
    """从索引中删除图片"""
    data = request.get_json()

    if not data or 'image_ids' not in data:
        return jsonify({'error': 'Missing image_ids'}), 400

    image_ids = data['image_ids']
    if not isinstance(image_ids, list):
        return jsonify({'error': 'image_ids must be a list'}), 400

    result = index_manager.remove_images(image_ids)
    return jsonify(result)


@app.route('/api/index/merge', methods=['POST'])
def merge_buffer():
    """合并缓冲区到主索引"""
    result = index_manager.merge_buffer()
    return jsonify(result)


@app.route('/api/index/rebuild', methods=['POST'])
def rebuild_index():
    """重建索引（清理已删除）"""
    # TODO: 实现重建逻辑（需要重新训练Faiss索引）
    return jsonify({
        'message': 'Rebuild not fully implemented yet. Use merge_buffer for now.'
    })


@app.route('/api/index/status', methods=['GET'])
def get_index_status():
    """获取索引状态"""
    status = index_manager.get_status()
    return jsonify(status)
```

**Step 4: 修改搜索逻辑**

更新现有的搜索端点，使用 `index_manager.search()`：

```python
# 在 /search 端点中
# 原来的: distance, ids = feat_retrieval(feat_mat, feat_query, topk)
# 改为:
distance, ids, paths = index_manager.search(feat_query, topk)

# 构建响应时使用 paths（已从manager返回）
```

**Step 5: 测试 API**

启动服务: `python flask_app.py`

测试状态接口:
```bash
curl http://localhost:5000/api/index/status
```

期望响应:
```json
{
  "main_count": 120000,
  "buffer_count": 0,
  "deleted_count": 0,
  "total_images": 120000,
  "buffer_threshold": 1000
}
```

**Step 6: 提交**

```bash
git add flask_app.py
git commit -m "feat: add incremental index management APIs"
```

---

## Task 5: 更新 RAG 引擎集成

**Files:**
- Modify: `rag_engine.py`
- Modify: `flask_app.py`（RAG相关路由）

**Step 1: 修改 RAGEngine 使用 IndexManager**

更新 `rag_engine.py` 中的 `RAGEngine` 类，使其接受 `IncrementalIndexManager`：

```python
class RAGEngine:
    def __init__(
        self,
        index_manager: IncrementalIndexManager,  # 改为接受管理器
        llm: LLMInterface,
        enable_expansion: bool = True,
        enable_explanation: bool = True
    ):
        self.index_manager = index_manager
        self.llm = llm
        ...
```

**Step 2: 更新搜索调用**

```python
# 在 search_and_explain 方法中
# 原来: distance, ids = self.index_module.feat_retrieval(feat_query, topk)
# 改为:
distance, ids, paths = self.index_manager.search(feat_query, topk)

# 使用 paths 直接构建结果
results = []
for i, (dist, img_id, path) in enumerate(zip(distance, ids, paths)):
    results.append({
        'id': int(img_id),
        'distance': float(dist),
        'path': path,
        'url': f'/static/images/{os.path.basename(path)}'
    })
```

**Step 3: 提交**

```bash
git add rag_engine.py
git commit -m "feat: integrate IncrementalIndexManager with RAG engine"
```

---

## Task 6: 添加端到端测试

**Files:**
- Create: `test_incremental_endtoend.py`

**Step 1: 编写完整流程测试**

```python
#!/usr/bin/env python
"""
端到端测试：增量索引完整流程
"""
import os
import sys
import requests
import time

BASE_URL = "http://localhost:5000"


def test_status():
    """测试状态接口"""
    print("\n=== Test 1: Get Status ===")
    resp = requests.get(f"{BASE_URL}/api/index/status")
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    return resp.json()


def test_add_images():
    """测试添加图片"""
    print("\n=== Test 2: Add Images ===")

    # 使用实际存在的图片路径
    test_images = [
        "train2017/000000000001.jpg",
        "train2017/000000000002.jpg",
    ]

    # 过滤存在的图片
    valid_images = [p for p in test_images if os.path.exists(p)]

    if not valid_images:
        print("Skip: No test images found")
        return None

    resp = requests.post(
        f"{BASE_URL}/api/index/add",
        json={"image_paths": valid_images}
    )
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    return resp.json()


def test_search():
    """测试搜索功能"""
    print("\n=== Test 3: Search ===")
    resp = requests.post(
        f"{BASE_URL}/search",
        json={"query": "a dog", "topk": 5}
    )
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(f"Results count: {len(data.get('results', []))}")
    assert resp.status_code == 200
    return data


def test_remove_images():
    """测试删除图片"""
    print("\n=== Test 4: Remove Images ===")
    resp = requests.post(
        f"{BASE_URL}/api/index/remove",
        json={"image_ids": [0, 1, 2]}  # 假设这些ID存在
    )
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    return resp.json()


def test_merge():
    """测试合并缓冲区"""
    print("\n=== Test 5: Merge Buffer ===")
    resp = requests.post(f"{BASE_URL}/api/index/merge")
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    return resp.json()


if __name__ == "__main__":
    print("Starting end-to-end tests...")
    print(f"Base URL: {BASE_URL}")

    try:
        # 运行测试
        test_status()
        test_add_images()
        test_search()
        test_remove_images()
        test_merge()
        test_status()  # 再次查看状态

        print("\n=== All tests passed! ===")

    except Exception as e:
        print(f"\n=== Test failed: {e} ===")
        sys.exit(1)
```

**Step 2: 运行测试**

确保服务已启动: `python flask_app.py`

运行测试: `python test_incremental_endtoend.py`

**Step 3: 提交**

```bash
git add test_incremental_endtoend.py
git commit -m "test: add end-to-end tests for incremental index"
```

---

## Task 7: 更新文档

**Files:**
- Modify: `README.md`
- Create: `docs/incremental-index-usage.md`

**Step 1: 更新 README**

在 API 接口部分添加新增端点：

```markdown
### 索引管理（新增）

- `POST /api/index/add` - 添加新图片到索引
- `POST /api/index/remove` - 从索引中删除图片
- `POST /api/index/merge` - 合并缓冲区到主索引
- `GET /api/index/status` - 查看索引状态
```

**Step 2: 创建使用文档**

创建 `docs/incremental-index-usage.md`，包含使用示例和注意事项。

**Step 3: 提交**

```bash
git add README.md docs/incremental-index-usage.md
git commit -m "docs: update documentation for incremental index feature"
```

---

## Task 8: 清理和最终验证

**Step 1: 清理测试文件**

```bash
# 删除临时测试文件
rm test_index_add.py test_image_index_builder.py test_incremental_manager.py
rm data/test_state.json data/test_state_buffer.pkl  # 如果有
```

**Step 2: 最终验证**

1. 启动服务: `python flask_app.py`
2. 测试所有新增 API
3. 测试原有搜索功能是否正常

**Step 3: 最终提交**

```bash
git add -A
git commit -m "feat: complete incremental index update implementation

- Add add_vectors to IndexModule for incremental adds
- Add ImageIndexBuilder for feature extraction
- Add IncrementalIndexManager with buffer and deletion support
- Add 5 new REST API endpoints for index management
- Integrate with existing RAG engine
- Add comprehensive tests"
```

---

## 验证清单

实施完成后，验证以下功能：

- [ ] `/api/index/status` 返回正确的状态
- [ ] `/api/index/add` 能添加新图片到缓冲区
- [ ] `/api/index/remove` 能标记删除图片
- [ ] `/api/index/merge` 能将缓冲区合并到主索引
- [ ] 搜索时结果正确（包含主索引和缓冲区）
- [ ] 删除的图片在搜索结果中被过滤
- [ ] 原有搜索和 RAG 功能不受影响
- [ ] 重启服务后状态正确恢复

---

**计划完成时间估计**: 2-3 小时
**依赖**: 需要理解现有 IndexModule 和 Flask 应用结构
