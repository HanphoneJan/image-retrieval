# -*- coding=utf-8 -*-
"""
# @file name  : base_config.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-04-30
@brief      : 基础配置参数，使用 dataclass 提供类型支持
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

# 加载.env文件
try:
    from dotenv import load_dotenv
    load_dotenv('.env')
except ImportError:
    pass

config_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(config_BASE_DIR)


@dataclass
class Config:
    """应用配置类"""

    # CLIP 模型配置
    clip_backbone_type: str = 'ViT-B/32'

    # 设备配置
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    # 路径配置
    image_file_dir: str = field(default_factory=lambda: os.path.join(
        PROJECT_ROOT, 'train2017'
    ))
    database_dir: str = field(default_factory=lambda: os.path.join(
        PROJECT_ROOT, 'data'
    ))

    # 特征和映射文件路径（动态计算）
    feat_mat_path: str = field(default="")
    map_dict_path: str = field(default="")

    # 索引配置
    index_string: str = 'IVF4096,PQ32x8'
    feat_dim: int = 512
    topk: int = 20

    # LLM / RAG 配置
    llm_base_url: str = field(default_factory=lambda: os.getenv('LLM_BASE_URL', ''))
    llm_api_key: str = field(default_factory=lambda: os.getenv('LLM_API_KEY', ''))
    llm_model: str = field(default_factory=lambda: os.getenv('LLM_MODEL', 'gpt-3.5-turbo'))

    # RAG 功能开关
    rag_enable_expansion: bool = True
    rag_enable_explanation: bool = True
    rag_max_expansions: int = 3
    rag_context_size: int = 5

    # Agent 工具配置
    agent_tools_enabled: bool = True

    def __post_init__(self):
        """初始化后处理：创建目录、计算动态路径等"""
        # 确保数据库目录存在
        if not os.path.exists(self.database_dir):
            os.makedirs(self.database_dir)

        # 根据模型类型确定特征维度
        if 'B' in self.clip_backbone_type:
            self.feat_dim = 512
        else:
            self.feat_dim = 768

        # 动态生成文件路径
        postfix = '_'.join(self.image_file_dir.split(os.path.sep)[1:])
        backbone_str = self.clip_backbone_type.replace('/', '_')

        self.feat_mat_path = os.path.join(
            self.database_dir, f'feat_mat-{postfix}-{backbone_str}.pkl'
        )
        self.map_dict_path = os.path.join(
            self.database_dir, f'map_dict-{postfix}-{backbone_str}.pkl'
        )


# 全局配置实例
CFG = Config()
