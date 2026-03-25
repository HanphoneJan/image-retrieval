# test_index_add.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from retrieval_by_faiss import IndexModule

# 创建测试数据 - 使用更多数据满足IVF要求
feat_mat = np.random.randn(5000, 512).astype(np.float32)

# 初始化索引 - 使用较小的nlist以适应测试数据量
index = IndexModule("IVF100,PQ32x8", 512, feat_mat)
print(f"Initial count: {index.get_total_count()}")

# 添加新向量
new_vectors = np.random.randn(10, 512).astype(np.float32)
new_ids = index.add_vectors(new_vectors)
print(f"Added IDs: {new_ids}")
print(f"New count: {index.get_total_count()}")

# 验证结果
assert index.get_total_count() == 5010, f"Expected 5010, got {index.get_total_count()}"
assert new_ids == list(range(5000, 5010)), f"Unexpected IDs: {new_ids}"
print("\n[OK] All tests passed!")
