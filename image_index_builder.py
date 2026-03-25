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
