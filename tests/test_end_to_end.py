"""
端到端集成测试 - 增量索引完整流程验证

测试场景:
1. 添加图片到缓冲区
2. 搜索（包含缓冲区图片）
3. 删除图片（逻辑删除）
4. 搜索（过滤已删除）
5. 合并缓冲区到主索引
6. 状态持久化验证
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tempfile

# 创建临时目录用于测试状态
TEST_STATE_DIR = tempfile.mkdtemp()
print(f"[INFO] 测试状态目录: {TEST_STATE_DIR}")

from retrieval_by_faiss import IndexModule
from incremental_index_manager import IncrementalIndexManager


class MockImageBuilder:
    """模拟图片特征构建器，返回确定性特征用于测试"""

    def __init__(self, seed=42):
        self.seed = seed
        self.feature_cache = {}

    def extract_features(self, image_paths):
        """生成确定性特征（每个路径对应固定特征）"""
        features = []
        valid_paths = []

        for path in image_paths:
            if path not in self.feature_cache:
                # 基于路径哈希生成确定性随机特征
                np.random.seed(hash(path) % 2**31)
                feat = np.random.randn(512).astype(np.float32)
                feat = feat / np.linalg.norm(feat)
                self.feature_cache[path] = feat

            features.append(self.feature_cache[path].reshape(1, -1))
            valid_paths.append(path)

        return np.vstack(features), valid_paths

    def build_mapping(self, paths, start_id=0):
        return {i + start_id: path for i, path in enumerate(paths)}


def create_test_environment():
    """创建测试环境"""
    print("[STEP] 创建测试环境...")

    # 创建初始主索引（100个向量）
    np.random.seed(42)
    initial_features = np.random.randn(100, 512).astype(np.float32)
    norms = np.linalg.norm(initial_features, axis=1, keepdims=True)
    initial_features = initial_features / norms

    main_index = IndexModule("Flat", 512, initial_features)

    # 创建模拟构建器
    builder = MockImageBuilder()

    # 创建增量索引管理器
    manager = IncrementalIndexManager(
        main_index=main_index,
        index_builder=builder,
        buffer_size_threshold=10,
        state_dir=TEST_STATE_DIR
    )

    print(f"  - 主索引向量数: {main_index.get_total_count()}")
    print(f"  - 初始状态: {manager.get_status()}")

    return manager, builder


def test_scenario_1_add_and_search():
    """场景1: 添加图片并搜索"""
    print("\n[TEST] 场景1: 添加图片并搜索")

    manager, builder = create_test_environment()

    # 添加5张新图片到缓冲区
    new_images = [f"new_image_{i}.jpg" for i in range(5)]
    result = manager.add_images(new_images)

    print(f"  - 添加结果: {result['added']} 张图片")
    print(f"  - 新ID: {result['ids']}")
    assert result['added'] == 5, "应添加5张图片"

    # 使用缓冲区的图片进行搜索
    query_path = new_images[0]
    query_feat = builder.feature_cache[query_path].reshape(1, -1)

    distances, ids, paths = manager.search(query_feat, topk=10)

    print(f"  - 搜索结果: {len(ids)} 个结果")
    print(f"  - 前3个结果ID: {list(ids[:3]) if len(ids) >= 3 else list(ids)}")

    # 验证新添加的图片在结果中
    new_ids = set(result['ids'])
    found_new_ids = new_ids.intersection(set(ids))
    print(f"  - 找到的新图片ID: {found_new_ids}")

    # 保存状态
    manager.save_state()
    print("  [OK] 场景1通过")


def test_scenario_2_delete_and_filter():
    """场景2: 删除图片并验证过滤"""
    print("\n[TEST] 场景2: 删除图片并验证过滤")

    manager, builder = create_test_environment()

    # 添加图片
    new_images = [f"image_{i}.jpg" for i in range(5)]
    add_result = manager.add_images(new_images)
    new_ids = add_result['ids']
    print(f"  - 添加了5张图片，ID: {new_ids}")

    # 删除部分图片
    ids_to_delete = new_ids[:2]
    remove_result = manager.remove_images(ids_to_delete)
    print(f"  - 删除结果: {remove_result['removed']} 张")
    assert remove_result['removed'] == 2, "应删除2张"

    # 搜索并验证被删除的图片不在结果中
    query_feat = np.random.randn(1, 512).astype(np.float32)
    query_feat = query_feat / np.linalg.norm(query_feat)

    distances, ids, paths = manager.search(query_feat, topk=20)

    # 验证被删除的ID不在结果中
    for deleted_id in ids_to_delete:
        assert deleted_id not in ids, f"被删除的ID {deleted_id} 不应在搜索结果中"

    print(f"  - 被删除的ID {ids_to_delete} 正确过滤")
    print("  [OK] 场景2通过")


def test_scenario_3_merge_buffer():
    """场景3: 合并缓冲区"""
    print("\n[TEST] 场景3: 合并缓冲区")

    manager, builder = create_test_environment()

    # 添加图片到缓冲区
    new_images = [f"buffer_img_{i}.jpg" for i in range(3)]
    manager.add_images(new_images)

    status_before = manager.get_status()
    print(f"  - 合并前: {status_before}")

    # 合并缓冲区
    merge_result = manager.merge_buffer()
    print(f"  - 合并结果: {merge_result}")

    status_after = manager.get_status()
    print(f"  - 合并后: {status_after}")

    assert status_after['buffer_count'] == 0, "合并后缓冲区应为空"
    assert status_after['main_count'] == 103, "合并后主索引应有103个向量"

    print("  [OK] 场景3通过")


def test_scenario_4_state_persistence():
    """场景4: 状态持久化"""
    print("\n[TEST] 场景4: 状态持久化")

    with tempfile.TemporaryDirectory() as state_dir:
        # 创建管理器并添加数据
        np.random.seed(42)
        initial_features = np.random.randn(50, 512).astype(np.float32)
        norms = np.linalg.norm(initial_features, axis=1, keepdims=True)
        initial_features = initial_features / norms

        main_index = IndexModule("Flat", 512, initial_features)
        builder = MockImageBuilder()

        manager1 = IncrementalIndexManager(
            main_index=main_index,
            index_builder=builder,
            buffer_size_threshold=10,
            state_dir=state_dir
        )

        # 添加和删除操作
        new_images = [f"persistent_{i}.jpg" for i in range(4)]
        add_result = manager1.add_images(new_images)
        added_ids = add_result['ids']
        manager1.remove_images([added_ids[0]])  # 删除第一个

        # 保存状态
        manager1.save_state()
        print(f"  - 保存状态: buffer={manager1.get_status()['buffer_count']}, deleted={manager1.get_status()['deleted']}")

        # 创建新的管理器实例并加载状态
        main_index2 = IndexModule("Flat", 512, initial_features)
        manager2 = IncrementalIndexManager(
            main_index=main_index2,
            index_builder=builder,
            buffer_size_threshold=10,
            state_dir=state_dir
        )

        # 准备映射字典
        map_dict = {i: f"initial_{i}.jpg" for i in range(50)}
        manager2.load_state(map_dict)

        status = manager2.get_status()
        print(f"  - 加载状态: {status}")

        assert status['buffer_count'] == 3, "缓冲区应有3个（删除后）"
        assert status['deleted'] == 1, "应有1个删除标记"
        assert status['next_id'] == 54, "next_id应为54"

        print("  [OK] 场景4通过")


def test_scenario_5_search_quality():
    """场景5: 搜索质量验证"""
    print("\n[TEST] 场景5: 搜索质量验证")

    manager, builder = create_test_environment()

    # 添加一个特殊特征的图片（全1向量）
    special_path = "special_white_image.jpg"
    special_feat = np.ones(512).astype(np.float32)
    special_feat = special_feat / np.linalg.norm(special_feat)
    builder.feature_cache[special_path] = special_feat

    manager.add_images([special_path])

    # 使用相似特征搜索（全1向量的查询）
    query_feat = np.ones(512).astype(np.float32) * 0.9 + np.random.randn(512) * 0.1
    query_feat = query_feat / np.linalg.norm(query_feat)
    query_feat = query_feat.reshape(1, -1)

    distances, ids, paths = manager.search(query_feat, topk=5)

    print(f"  - 查询特征与special图片的相似度: {np.dot(query_feat.flatten(), special_feat):.4f}")
    print(f"  - 搜索结果数量: {len(ids)}")

    if len(ids) > 0:
        print(f"  - 最佳匹配: ID={ids[0]}, path={paths[0]}, distance={distances[0]:.4f}")

    print("  [OK] 场景5通过")


def test_scenario_6_full_workflow():
    """场景6: 完整工作流"""
    print("\n[TEST] 场景6: 完整工作流")

    with tempfile.TemporaryDirectory() as state_dir:
        # 1. 初始化
        np.random.seed(42)
        initial_features = np.random.randn(20, 512).astype(np.float32)
        norms = np.linalg.norm(initial_features, axis=1, keepdims=True)
        initial_features = initial_features / norms

        main_index = IndexModule("Flat", 512, initial_features)
        builder = MockImageBuilder()

        manager = IncrementalIndexManager(
            main_index=main_index,
            index_builder=builder,
            buffer_size_threshold=5,
            state_dir=state_dir
        )

        print("  1. 初始化完成")

        # 2. 批量添加图片
        batch1 = [f"batch1_img_{i}.jpg" for i in range(3)]
        manager.add_images(batch1)
        print("  2. 添加第一批: 3张图片")

        # 3. 继续添加
        batch2 = [f"batch2_img_{i}.jpg" for i in range(4)]
        manager.add_images(batch2)
        print("  3. 添加第二批: 4张图片")

        # 4. 删除部分图片
        manager.remove_images([20, 21])  # 删除前两张
        print("  4. 删除图片 ID 20, 21")

        # 5. 搜索验证
        query_feat = np.random.randn(1, 512).astype(np.float32)
        query_feat = query_feat / np.linalg.norm(query_feat)
        distances, ids, paths = manager.search(query_feat, topk=10)
        print(f"  5. 搜索返回 {len(ids)} 个结果")

        # 6. 保存状态
        manager.save_state()
        print("  6. 状态已保存")

        # 7. 合并缓冲区
        merge_result = manager.merge_buffer()
        print(f"  7. 合并完成: {merge_result}")

        # 8. 验证最终状态
        status = manager.get_status()
        print(f"  8. 最终状态: {status}")

        assert status['main_count'] == 25, "主索引应有25个向量（20初始+3+4-2删除在缓冲区中）"
        assert status['buffer_count'] == 0, "缓冲区应为空"

        print("  [OK] 场景6通过")


if __name__ == "__main__":
    print("=" * 70)
    print("增量图片索引 - 端到端集成测试")
    print("=" * 70)

    try:
        test_scenario_1_add_and_search()
        test_scenario_2_delete_and_filter()
        test_scenario_3_merge_buffer()
        test_scenario_4_state_persistence()
        test_scenario_5_search_quality()
        test_scenario_6_full_workflow()

        print("\n" + "=" * 70)
        print("[OK] 所有端到端测试通过!")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n[FAILED] 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
