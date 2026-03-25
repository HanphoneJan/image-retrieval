# incremental_index_manager.py
"""
增量索引管理器 - 核心增量管理逻辑

职责：
1. 管理主索引和缓冲区
2. 协调添加/删除/搜索操作
3. 状态持久化
"""

import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Set
from pathlib import Path

from retrieval_by_faiss import IndexModule
from image_index_builder import ImageIndexBuilder


class IncrementalIndexManager:
    """
    增量索引管理器

    核心设计：
    - 主索引（Faiss IVF）：存储已训练的大规模数据
    - 缓冲区（Python列表）：存储新添加的小规模数据
    - 删除标记（Set）：逻辑删除，重建时清理
    """

    def __init__(self,
                 main_index: IndexModule,
                 index_builder: ImageIndexBuilder,
                 buffer_size_threshold: int = 1000,
                 state_dir: str = "./data/index_state"):
        """
        初始化增量索引管理器

        Args:
            main_index: 主索引模块（已训练的Faiss索引）
            index_builder: 图片索引构建器（用于提取新图片特征）
            buffer_size_threshold: 缓冲区大小阈值，超过则建议合并
            state_dir: 状态保存目录
        """
        self.main_index = main_index
        self.index_builder = index_builder
        self.buffer_size_threshold = buffer_size_threshold
        self.state_dir = Path(state_dir)

        # 缓冲区存储
        self.buffer_features: List[np.ndarray] = []  # 特征向量列表
        self.buffer_paths: List[str] = []            # 路径列表
        self.buffer_ids: List[int] = []              # 分配的ID列表

        # 删除标记
        self.deleted_ids: Set[int] = set()

        # ID分配计数器（从主索引当前数量开始）
        self.next_id = main_index.get_total_count()

        # 映射字典（动态构建）
        self.id_to_path: Dict[int, str] = {}

        # 确保状态目录存在
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def add_images(self, image_paths: List[str]) -> Dict:
        """
        添加新图片到缓冲区

        Args:
            image_paths: 图片路径列表

        Returns:
            {"added": int, "ids": List[int], "failed": List[str]}
        """
        if not image_paths:
            return {"added": 0, "ids": [], "failed": []}

        # 提取特征
        features, valid_paths = self.index_builder.extract_features(image_paths)

        if len(features) == 0:
            return {"added": 0, "ids": [], "failed": image_paths}

        # 分配新ID
        new_ids = list(range(self.next_id, self.next_id + len(valid_paths)))
        self.next_id += len(valid_paths)

        # 存入缓冲区
        for i, (feat, path, id_) in enumerate(zip(features, valid_paths, new_ids)):
            self.buffer_features.append(feat.reshape(1, -1))  # 保持 (1, 512) 形状
            self.buffer_paths.append(path)
            self.buffer_ids.append(id_)
            self.id_to_path[id_] = path

        # 计算失败的图片
        failed_paths = list(set(image_paths) - set(valid_paths))

        result = {
            "added": len(valid_paths),
            "ids": new_ids,
            "failed": failed_paths
        }

        # 检查是否需要提示合并
        if len(self.buffer_ids) >= self.buffer_size_threshold:
            print(f"[INFO] 缓冲区已达到阈值 ({len(self.buffer_ids)}/{self.buffer_size_threshold})，建议执行 merge_buffer()")

        return result

    def remove_images(self, image_ids: List[int]) -> Dict:
        """
        标记删除图片（逻辑删除）

        Args:
            image_ids: 要删除的图片ID列表

        Returns:
            {"removed": int, "not_found": List[int]}
        """
        removed_count = 0
        not_found = []

        for id_ in image_ids:
            # 检查ID是否存在（主索引或缓冲区）
            exists_in_main = id_ < self.main_index.get_total_count()
            exists_in_buffer = id_ in self.buffer_ids

            if not exists_in_main and not exists_in_buffer:
                not_found.append(id_)
                continue

            # 标记为删除
            self.deleted_ids.add(id_)
            removed_count += 1

            # 如果在缓冲区中，也可以立即移除（可选优化）
            if id_ in self.buffer_ids:
                idx = self.buffer_ids.index(id_)
                self.buffer_features.pop(idx)
                self.buffer_paths.pop(idx)
                self.buffer_ids.pop(idx)

        return {"removed": removed_count, "not_found": not_found}

    def search(self, query_vector: np.ndarray, topk: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        搜索（合并主索引+缓冲区结果）

        Args:
            query_vector: 查询向量，shape (1, 512)
            topk: 返回结果数量

        Returns:
            (distances, ids, paths): 距离数组、ID数组、路径列表
        """
        # 搜索主索引
        main_distances, main_ids = self._search_main_index(query_vector, topk * 2)

        # 搜索缓冲区
        buffer_distances, buffer_ids, buffer_paths = self._search_buffer(query_vector, topk * 2)

        # 合并结果
        all_distances = []
        all_ids = []
        all_paths = []

        # 添加主索引结果
        if len(main_ids) > 0:
            # 过滤已删除的ID
            for dist, id_ in zip(main_distances, main_ids):
                if id_ not in self.deleted_ids:
                    all_distances.append(dist)
                    all_ids.append(id_)
                    all_paths.append(self._get_path_by_id(id_))

        # 添加缓冲区结果
        for dist, id_, path in zip(buffer_distances, buffer_ids, buffer_paths):
            if id_ not in self.deleted_ids:
                all_distances.append(dist)
                all_ids.append(id_)
                all_paths.append(path)

        # 按距离排序并取topk
        if len(all_distances) == 0:
            return np.array([]), np.array([]), []

        sorted_indices = np.argsort(all_distances)[:topk]

        final_distances = np.array([all_distances[i] for i in sorted_indices])
        final_ids = np.array([all_ids[i] for i in sorted_indices])
        final_paths = [all_paths[i] for i in sorted_indices]

        return final_distances, final_ids, final_paths

    def _search_main_index(self, query_vector: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        """搜索主索引"""
        if self.main_index.get_total_count() == 0:
            return np.array([]), np.array([])

        distances, ids = self.main_index.feat_retrieval(query_vector, topk)
        return distances, ids

    def _search_buffer(self, query_vector: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """搜索缓冲区（暴力搜索）"""
        if len(self.buffer_features) == 0:
            return np.array([]), np.array([]), []

        # 合并缓冲区特征
        buffer_matrix = np.vstack(self.buffer_features)  # (N, 512)

        # 计算余弦相似度（向量已归一化，点积即余弦相似度）
        similarities = np.dot(buffer_matrix, query_vector.T).flatten()  # (N,)

        # 转换为距离（1 - similarity）
        distances = 1 - similarities

        # 取topk
        topk = min(topk, len(distances))
        top_indices = np.argsort(distances)[:topk]

        result_distances = distances[top_indices]
        result_ids = np.array([self.buffer_ids[i] for i in top_indices])
        result_paths = [self.buffer_paths[i] for i in top_indices]

        return result_distances, result_ids, result_paths

    def _get_path_by_id(self, id_: int) -> str:
        """根据ID获取图片路径"""
        # 先查映射字典
        if id_ in self.id_to_path:
            return self.id_to_path[id_]

        # 再查缓冲区
        if id_ in self.buffer_ids:
            idx = self.buffer_ids.index(id_)
            return self.buffer_paths[idx]

        return ""

    def merge_buffer(self) -> Dict:
        """
        将缓冲区合并到主索引

        注意：合并后主索引需要重新训练，因此这会触发重建

        Returns:
            {"merged": int, "total": int}
        """
        if len(self.buffer_features) == 0:
            return {"merged": 0, "total": self.main_index.get_total_count()}

        # 合并特征
        buffer_matrix = np.vstack(self.buffer_features).astype(np.float32)

        # 直接添加到主索引（Faiss IVF支持add操作）
        new_ids = self.main_index.add_vectors(buffer_matrix)

        merged_count = len(new_ids)

        # 清空缓冲区
        self.buffer_features = []
        self.buffer_paths = []
        self.buffer_ids = []

        return {"merged": merged_count, "total": self.main_index.get_total_count()}

    def get_status(self) -> Dict:
        """
        获取当前索引状态

        Returns:
            {
                "main_count": int,      # 主索引中的向量数
                "buffer_count": int,    # 缓冲区中的向量数
                "deleted": int,         # 已删除的ID数量
                "next_id": int          # 下一个分配的ID
            }
        """
        return {
            "main_count": self.main_index.get_total_count(),
            "buffer_count": len(self.buffer_ids),
            "deleted": len(self.deleted_ids),
            "next_id": self.next_id
        }

    def save_state(self) -> str:
        """
        保存当前状态到磁盘

        Returns:
            保存的状态文件路径
        """
        # 保存状态JSON
        state = {
            "next_id": self.next_id,
            "deleted_ids": list(self.deleted_ids),
            "buffer_ids": self.buffer_ids,
            "buffer_paths": self.buffer_paths,
        }

        state_path = self.state_dir / "index_state.json"
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        # 保存缓冲区特征
        if len(self.buffer_features) > 0:
            buffer_matrix = np.vstack(self.buffer_features)
            buffer_path = self.state_dir / "buffer_features.pkl"
            with open(buffer_path, 'wb') as f:
                pickle.dump(buffer_matrix, f)

        return str(state_path)

    def load_state(self, map_dict: Dict[int, str]) -> None:
        """
        从磁盘加载状态

        Args:
            map_dict: ID到路径的映射字典（主索引）
        """
        self.id_to_path = map_dict.copy()

        state_path = self.state_dir / "index_state.json"
        if not state_path.exists():
            print("[INFO] 没有找到状态文件，使用空状态")
            return

        # 加载状态JSON
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        self.next_id = state.get("next_id", self.main_index.get_total_count())
        self.deleted_ids = set(state.get("deleted_ids", []))
        self.buffer_ids = state.get("buffer_ids", [])
        self.buffer_paths = state.get("buffer_paths", [])

        # 加载缓冲区特征
        buffer_path = self.state_dir / "buffer_features.pkl"
        if buffer_path.exists() and len(self.buffer_ids) > 0:
            with open(buffer_path, 'rb') as f:
                buffer_matrix = pickle.load(f)

            # 还原为列表
            self.buffer_features = [
                buffer_matrix[i:i+1]  # 保持 (1, 512) 形状
                for i in range(len(self.buffer_ids))
            ]

        print(f"[INFO] 状态已加载: {len(self.buffer_ids)} 个缓冲区项, {len(self.deleted_ids)} 个删除标记")

    def rebuild_index(self, all_features: np.ndarray, all_paths: List[str], index_string: str) -> IndexModule:
        """
        完全重建索引（清理已删除的数据）

        Args:
            all_features: 所有特征矩阵
            all_paths: 所有路径列表
            index_string: Faiss索引字符串

        Returns:
            新的IndexModule实例
        """
        # 过滤已删除的ID
        valid_indices = [
            i for i in range(len(all_paths))
            if i not in self.deleted_ids
        ]

        # 提取有效数据
        valid_features = all_features[valid_indices]
        valid_paths = [all_paths[i] for i in valid_indices]

        # 创建新索引
        new_index = IndexModule(index_string, valid_features.shape[1], valid_features)

        # 清空删除标记和缓冲区
        self.deleted_ids.clear()
        self.buffer_features = []
        self.buffer_paths = []
        self.buffer_ids = []
        self.next_id = new_index.get_total_count()

        # 更新映射
        self.id_to_path = {i: path for i, path in enumerate(valid_paths)}
        self.main_index = new_index

        print(f"[INFO] 索引重建完成: {len(valid_paths)} 个有效向量")

        return new_index
