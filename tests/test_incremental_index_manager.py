"""
测试 IncrementalIndexManager 功能
- 添加图片
- 删除图片
- 搜索（主索引+缓冲区）
- 状态保存/加载
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tempfile

# 模拟图片路径（不需要真实图片，因为我们直接测试向量操作）
from incremental_index_manager import IncrementalIndexManager
from retrieval_by_faiss import IndexModule


def create_mock_builder():
    """创建一个模拟的 ImageIndexBuilder 用于测试"""
    class MockBuilder:
        def extract_features(self, image_paths):
            """模拟特征提取 - 返回随机特征"""
            features = np.random.randn(len(image_paths), 512).astype(np.float32)
            # L2归一化
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / norms
            return features, image_paths

        def build_mapping(self, paths, start_id=0):
            return {i + start_id: path for i, path in enumerate(paths)}

    return MockBuilder()


def test_add_images():
    """测试添加图片到缓冲区"""
    print("测试: 添加图片到缓冲区...")

    # 创建主索引
    initial_features = np.random.randn(100, 512).astype(np.float32)
    main_index = IndexModule("Flat", 512, initial_features)

    # 创建管理器
    manager = IncrementalIndexManager(
        main_index=main_index,
        index_builder=create_mock_builder(),
        buffer_size_threshold=10
    )

    # 添加图片
    paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    result = manager.add_images(paths)

    # 验证结果
    assert result["added"] == 3, f"期望添加3个, 实际 {result['added']}"
    assert len(result["ids"]) == 3, "期望返回3个ID"
    assert result["ids"] == [100, 101, 102], "ID应从100开始"
    assert result["failed"] == [], "没有失败的图片"

    # 验证状态
    status = manager.get_status()
    assert status["buffer_count"] == 3, "缓冲区应有3个"
    assert status["next_id"] == 103, "next_id 应为103"

    print("  [OK] 添加图片测试通过")


def test_remove_images():
    """测试删除图片（逻辑删除）"""
    print("测试: 删除图片...")

    # 创建主索引
    initial_features = np.random.randn(100, 512).astype(np.float32)
    main_index = IndexModule("Flat", 512, initial_features)

    # 创建管理器
    manager = IncrementalIndexManager(
        main_index=main_index,
        index_builder=create_mock_builder(),
        buffer_size_threshold=10
    )

    # 添加一些图片
    paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    _ = manager.add_images(paths)

    # 删除一个
    remove_result = manager.remove_images([100, 101])
    assert remove_result["removed"] == 2, "期望删除2个"

    # 验证状态
    status = manager.get_status()
    assert status["deleted"] == 2, "应有2个删除标记"
    assert status["buffer_count"] == 1, "缓冲区应只剩1个（即时移除）"

    print("  [OK] 删除图片测试通过")


def test_search():
    """测试搜索功能"""
    print("测试: 搜索功能...")

    # 创建主索引（使用简单索引便于测试）
    np.random.seed(42)
    initial_features = np.random.randn(10, 512).astype(np.float32)
    # 归一化
    norms = np.linalg.norm(initial_features, axis=1, keepdims=True)
    initial_features = initial_features / norms

    main_index = IndexModule("Flat", 512, initial_features)

    # 创建管理器
    manager = IncrementalIndexManager(
        main_index=main_index,
        index_builder=create_mock_builder(),
        buffer_size_threshold=10
    )

    # 添加缓冲区图片
    paths = ["buffer1.jpg", "buffer2.jpg"]
    manager.add_images(paths)

    # 构造查询向量（随机）
    query = np.random.randn(1, 512).astype(np.float32)
    query = query / np.linalg.norm(query)

    # 搜索
    distances, ids, paths_result = manager.search(query, topk=5)

    # 验证返回结果
    assert len(distances) <= 5, "返回结果不应超过topk"
    assert len(ids) == len(distances), "ids和distances长度应一致"
    assert len(paths_result) == len(distances), "paths和distances长度应一致"

    print(f"  搜索结果: {len(ids)} 个结果")
    print("  [OK] 搜索测试通过")


def test_search_with_deleted():
    """测试搜索时过滤已删除的ID"""
    print("测试: 搜索过滤已删除...")

    # 创建主索引
    np.random.seed(42)
    initial_features = np.random.randn(5, 512).astype(np.float32)
    norms = np.linalg.norm(initial_features, axis=1, keepdims=True)
    initial_features = initial_features / norms

    main_index = IndexModule("Flat", 512, initial_features)

    # 创建管理器
    manager = IncrementalIndexManager(
        main_index=main_index,
        index_builder=create_mock_builder(),
        buffer_size_threshold=10
    )

    # 删除主索引中的ID 0和1
    manager.remove_images([0, 1])

    # 构造查询向量
    query = np.random.randn(1, 512).astype(np.float32)
    query = query / np.linalg.norm(query)

    # 搜索
    distances, ids, paths_result = manager.search(query, topk=10)

    # 验证已删除的ID不在结果中
    assert 0 not in ids, "ID 0 已被删除，不应出现在结果中"
    assert 1 not in ids, "ID 1 已被删除，不应出现在结果中"

    print(f"  搜索结果: {list(ids)} (不包含0和1)")
    print("  [OK] 搜索过滤测试通过")


def test_merge_buffer():
    """测试合并缓冲区"""
    print("测试: 合并缓冲区...")

    # 创建主索引
    initial_features = np.random.randn(10, 512).astype(np.float32)
    main_index = IndexModule("Flat", 512, initial_features)

    # 创建管理器
    manager = IncrementalIndexManager(
        main_index=main_index,
        index_builder=create_mock_builder(),
        buffer_size_threshold=10
    )

    # 添加缓冲区图片
    paths = ["buffer1.jpg", "buffer2.jpg"]
    manager.add_images(paths)

    # 验证初始状态
    status = manager.get_status()
    assert status["buffer_count"] == 2
    assert status["main_count"] == 10

    # 合并缓冲区
    merge_result = manager.merge_buffer()

    # 验证合并结果
    assert merge_result["merged"] == 2, "期望合并2个"
    assert merge_result["total"] == 12, "合并后总数应为12"

    # 验证缓冲区已清空
    status = manager.get_status()
    assert status["buffer_count"] == 0, "缓冲区应为空"
    assert status["main_count"] == 12, "主索引应有12个"

    print("  [OK] 合并缓冲区测试通过")


def test_save_load_state():
    """测试状态保存和加载"""
    print("测试: 状态保存和加载...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建主索引
        initial_features = np.random.randn(10, 512).astype(np.float32)
        main_index = IndexModule("Flat", 512, initial_features)

        # 创建管理器并添加数据
        manager = IncrementalIndexManager(
            main_index=main_index,
            index_builder=create_mock_builder(),
            buffer_size_threshold=10,
            state_dir=tmpdir
        )

        # 添加缓冲区图片
        paths = ["buffer1.jpg", "buffer2.jpg", "buffer3.jpg"]
        manager.add_images(paths)
        manager.remove_images([10])  # 删除一个

        # 保存状态
        state_path = manager.save_state()
        assert os.path.exists(state_path), "状态文件应存在"

        # 创建新的管理器并加载状态
        main_index2 = IndexModule("Flat", 512, initial_features)
        manager2 = IncrementalIndexManager(
            main_index=main_index2,
            index_builder=create_mock_builder(),
            buffer_size_threshold=10,
            state_dir=tmpdir
        )

        # 加载状态
        manager2.load_state({0: "old1.jpg", 1: "old2.jpg"})

        # 验证加载的数据
        status = manager2.get_status()
        assert status["buffer_count"] == 2, "缓冲区应有2个（删除后剩余）"
        assert status["deleted"] == 1, "应有1个删除标记"
        assert status["next_id"] == 13, "next_id 应为13"

    print("  [OK] 状态保存加载测试通过")


def test_empty_operations():
    """测试空操作"""
    print("测试: 空操作...")

    # 创建主索引
    initial_features = np.random.randn(5, 512).astype(np.float32)
    main_index = IndexModule("Flat", 512, initial_features)

    manager = IncrementalIndexManager(
        main_index=main_index,
        index_builder=create_mock_builder(),
        buffer_size_threshold=10
    )

    # 空添加
    result = manager.add_images([])
    assert result["added"] == 0

    # 空删除
    result = manager.remove_images([])
    assert result["removed"] == 0

    # 空搜索
    query = np.random.randn(1, 512).astype(np.float32)
    query = query / np.linalg.norm(query)
    distances, ids, paths = manager.search(query, topk=5)
    # 应该返回主索引结果

    # 空合并
    result = manager.merge_buffer()
    assert result["merged"] == 0

    print("  [OK] 空操作测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("IncrementalIndexManager 测试套件")
    print("=" * 60)

    test_add_images()
    test_remove_images()
    test_search()
    test_search_with_deleted()
    test_merge_buffer()
    test_save_load_state()
    test_empty_operations()

    print("=" * 60)
    print("[OK] 所有测试通过!")
    print("=" * 60)
