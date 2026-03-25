"""
测试 ImageIndexBuilder 功能
- 特征提取
- 映射构建
- 错误处理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import tempfile

from image_index_builder import ImageIndexBuilder


def create_test_image(path, size=(224, 224), color=(255, 0, 0)):
    """创建一个测试图片"""
    img = Image.new('RGB', size, color)
    img.save(path)


def test_extract_features():
    """测试特征提取功能"""
    print("测试: 特征提取...")

    # 创建临时目录和图片
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试图片
        img1_path = os.path.join(tmpdir, "test1.jpg")
        img2_path = os.path.join(tmpdir, "test2.jpg")
        create_test_image(img1_path, color=(255, 0, 0))  # 红色
        create_test_image(img2_path, color=(0, 255, 0))  # 绿色

        # 测试特征提取
        builder = ImageIndexBuilder()
        features, valid_paths = builder.extract_features([img1_path, img2_path])

        # 验证结果
        assert features.shape == (2, 512), f"期望形状 (2, 512), 实际 {features.shape}"
        assert len(valid_paths) == 2, f"期望2个有效路径, 实际 {len(valid_paths)}"
        assert img1_path in valid_paths, "img1_path 应该在有效路径中"
        assert img2_path in valid_paths, "img2_path 应该在有效路径中"

        # 验证特征是否归一化 (L2范数应接近1)
        norms = np.linalg.norm(features, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), "特征向量应该已L2归一化"

    print("  [OK] 特征提取测试通过")


def test_extract_features_with_invalid_path():
    """测试处理无效路径"""
    print("测试: 无效路径处理...")

    builder = ImageIndexBuilder()

    # 创建临时目录和一个有效图片
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "valid.jpg")
        create_test_image(img_path)

        # 混合有效和无效路径
        invalid_path = "/nonexistent/path/image.jpg"
        features, valid_paths = builder.extract_features([invalid_path, img_path])

        # 验证结果
        assert features.shape == (1, 512), f"期望形状 (1, 512), 实际 {features.shape}"
        assert len(valid_paths) == 1, f"期望1个有效路径, 实际 {len(valid_paths)}"
        assert valid_paths[0] == img_path, "有效路径应该被保留"

    print("  [OK] 无效路径处理测试通过")


def test_build_mapping():
    """测试映射构建功能"""
    print("测试: 映射构建...")

    builder = ImageIndexBuilder()

    paths = ["/path/to/img1.jpg", "/path/to/img2.jpg", "/path/to/img3.jpg"]

    # 测试从0开始
    mapping = builder.build_mapping(paths, start_id=0)
    assert mapping == {0: "/path/to/img1.jpg", 1: "/path/to/img2.jpg", 2: "/path/to/img3.jpg"}

    # 测试从100开始
    mapping = builder.build_mapping(paths, start_id=100)
    assert mapping == {100: "/path/to/img1.jpg", 101: "/path/to/img2.jpg", 102: "/path/to/img3.jpg"}

    print("  [OK] 映射构建测试通过")


def test_empty_features():
    """测试空特征处理"""
    print("测试: 空特征处理...")

    builder = ImageIndexBuilder()
    features, valid_paths = builder.extract_features([])

    assert features.shape == (0, 512), f"期望形状 (0, 512), 实际 {features.shape}"
    assert valid_paths == [], "空列表应该返回空的有效路径"

    print("  [OK] 空特征处理测试通过")


def test_all_invalid_paths():
    """测试全部无效路径"""
    print("测试: 全部无效路径...")

    builder = ImageIndexBuilder()
    features, valid_paths = builder.extract_features([
        "/nonexistent/1.jpg",
        "/nonexistent/2.jpg"
    ])

    assert features.shape == (0, 512), f"期望形状 (0, 512), 实际 {features.shape}"
    assert valid_paths == [], "全部无效时应返回空列表"

    print("  [OK] 全部无效路径测试通过")


if __name__ == "__main__":
    print("=" * 50)
    print("ImageIndexBuilder 测试套件")
    print("=" * 50)

    test_extract_features()
    test_extract_features_with_invalid_path()
    test_build_mapping()
    test_empty_features()
    test_all_invalid_paths()

    print("=" * 50)
    print("[OK] 所有测试通过!")
    print("=" * 50)
