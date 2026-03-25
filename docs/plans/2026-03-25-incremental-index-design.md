# 增量图片索引更新设计文档

**日期**: 2026-03-25
**主题**: Incremental Image Index Update
**状态**: 已批准，待实施

---

## 1. 背景与问题

### 1.1 当前问题

当前 `IndexModule` 存在以下限制：
- 初始化时一次性训练和添加所有数据
- 没有提供增量添加接口
- 删除操作需要完全重建索引（Faiss IVF 不支持直接删除）
- 新增/删除单张图片需要重新运行 `image_feature_extract.py`（耗时2-4小时）

### 1.2 目标

实现支持增量添加和逻辑删除的图片索引管理系统，避免全量重建。

---

## 2. 设计方案

### 2.1 核心策略：带缓冲区的增量管理（方案B）

选择平衡实现复杂度和实用性的中间方案：
- **添加**：新图片先进入缓冲区，定期/手动合并到主索引
- **删除**：逻辑标记，搜索时过滤，重建时清理
- **搜索**：合并主索引和缓冲区结果

### 2.2 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                  IncrementalIndexManager                     │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │   Main Index    │  │  Buffer Index   │                   │
│  │  (Faiss IVF)    │  │  (Faiss Flat)   │                   │
│  │  - 已训练数据   │  │  - 新添加数据   │                   │
│  │  - 大规模       │  │  - 小规模       │                   │
│  └────────┬────────┘  └────────┬────────┘                   │
│           │                    │                            │
│           └────────┬───────────┘                            │
│                    ↓                                        │
│           ┌─────────────────┐                               │
│           │  Search Merger  │                               │
│           │  (结果合并+过滤) │                               │
│           └─────────────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 数据存储结构

```
data/
├── feat_mat.pkl              # 主特征矩阵
├── map_dict.pkl              # 主ID映射
├── index_state.json          # 索引状态（删除标记、缓冲区信息）
└── buffer/
    ├── buffer_feat_mat.pkl   # 缓冲区特征
    └── buffer_map_dict.pkl   # 缓冲区映射
```

---

## 3. 模块设计

### 3.1 IndexModule 增强

```python
class IndexModule:
    # 现有方法保持不变...

    def add_vectors(self, vectors: np.ndarray) -> List[int]:
        """增量添加向量到索引"""

    def remove_ids(self, ids: List[int]) -> None:
        """标记删除指定ID（逻辑删除）"""

    def get_trained_count(self) -> int:
        """获取已训练数据量"""
```

### 3.2 新增 IncrementalIndexManager

```python
class IncrementalIndexManager:
    """
    增量索引管理器

    职责：
    1. 管理主索引和缓冲区
    2. 协调添加/删除/搜索操作
    3. 状态持久化
    """

    def __init__(self,
                 main_index: IndexModule,
                 buffer_size_threshold: int = 1000):
        self.main_index = main_index
        self.buffer = []  # 缓冲区
        self.buffer_size_threshold = buffer_size_threshold
        self.deleted_ids = set()  # 已删除ID集合

    def add_images(self, image_paths: List[str]) -> Dict:
        """添加新图片到缓冲区"""

    def remove_images(self, image_ids: List[int]) -> None:
        """标记删除图片"""

    def search(self, query_vector: np.ndarray, topk: int) -> Tuple:
        """搜索（合并主索引+缓冲区）"""

    def merge_buffer(self) -> None:
        """将缓冲区合并到主索引"""

    def rebuild_index(self) -> None:
        """重建索引（清理已删除）"""

    def save_state(self, path: str) -> None:
        """保存当前状态"""

    def load_state(self, path: str) -> None:
        """加载状态"""
```

### 3.3 新增 ImageIndexBuilder

```python
class ImageIndexBuilder:
    """
    图片索引构建器

    职责：
    1. 提取图片CLIP特征
    2. 管理ID映射
    """

    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """批量提取图片特征"""

    def build_mapping(self, image_paths: List[str], start_id: int) -> Dict:
        """构建ID到路径的映射"""
```

---

## 4. API 设计

### 4.1 新增 REST API 端点

| 端点 | 方法 | 请求体 | 响应 | 功能 |
|:---|:---:|:---|:---|:---|
| `/api/index/add` | POST | `{"image_paths": ["path1", "path2"]}` | `{"added": 2, "ids": [1001, 1002]}` | 添加图片 |
| `/api/index/remove` | POST | `{"image_ids": [1001, 1002]}` | `{"removed": 2}` | 删除图片 |
| `/api/index/merge` | POST | - | `{"merged": 1000}` | 合并缓冲区 |
| `/api/index/rebuild` | POST | - | `{"total": 5000}` | 重建索引 |
| `/api/index/status` | GET | - | `{"main_count": 4000, "buffer_count": 100, "deleted": 50}` | 查看状态 |

### 4.2 与现有搜索 API 集成

现有 `/search` 和 `/api/search/rag` 端点无需修改，内部使用 `IncrementalIndexManager.search()` 替代直接调用 `IndexModule`。

---

## 5. 关键流程

### 5.1 添加图片流程

```
用户请求 /api/index/add
    ↓
验证图片路径存在
    ↓
提取CLIP特征（批量）
    ↓
存入缓冲区
    ↓
更新映射字典
    ↓
返回新分配的ID
    ↓
检查是否超过阈值
    ↓
是 → 自动触发合并（可选）
```

### 5.2 搜索流程

```
用户查询
    ↓
查询向量化（CLIP）
    ↓
搜索主索引 → 获得候选集A
搜索缓冲区 → 获得候选集B
    ↓
合并候选集（按距离排序）
    ↓
过滤已删除ID
    ↓
返回TopK结果
```

### 5.3 重建索引流程

```
触发重建
    ↓
加载主特征矩阵
    ↓
移除已删除ID对应的特征
    ↓
合并缓冲区特征
    ↓
重新训练Faiss索引
    ↓
保存新的特征矩阵和映射
    ↓
清空缓冲区
    ↓
清空已删除标记
```

---

## 6. 错误处理

| 场景 | 处理策略 |
|:---|:---|
| 图片路径不存在 | 跳过该图片，记录警告，返回部分成功 |
| 特征提取失败 | 跳过该图片，记录错误 |
| 索引保存失败 | 回滚到上次成功状态，返回错误 |
| 缓冲区合并失败 | 保留缓冲区，主索引不变，返回错误 |

---

## 7. 性能考虑

| 指标 | 目标 |
|:---|:---|
| 单张图片添加 | < 100ms（特征提取） |
| 缓冲区搜索 | < 50ms（数据量小） |
| 合并1000张 | < 5s |
| 重建索引 | 与初始训练同量级（2-4小时/12万图片） |

---

## 8. 测试策略

1. **单元测试**：每个新增类的方法
2. **集成测试**：添加→搜索→删除→重建完整流程
3. **性能测试**：大规模数据下的添加和搜索性能

---

## 9. 实现顺序

1. 增强 `IndexModule`（添加 `add_vectors` 等方法）
2. 实现 `ImageIndexBuilder`（特征提取封装）
3. 实现 `IncrementalIndexManager`（核心逻辑）
4. 添加 Flask API 端点
5. 编写测试
6. 更新文档

---

**批准人**: user
**批准日期**: 2026-03-25
