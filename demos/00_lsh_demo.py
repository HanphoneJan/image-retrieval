# -*- coding:utf-8 -*-
"""
@file name  : 00_lsh_demo.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-04-18
@brief      : LSH(Locality-Sensitive Hashing) 算法demo，有助于理解哈希函数（高斯分布）
              LSH是一种近似最近邻搜索算法，通过设计对局部敏感的哈希函数，
              使相似的数据点有更高概率映射到相同的哈希桶中，非相似数据点映射到
              相同桶的概率较低，从而高效地实现近似搜索
"""
import numpy as np  # 用于数值计算和数组操作
import random  # 用于生成随机数


def getHash(v, x, b, w):
    """
    获取单个数据点的哈希值，实现LSH中的基础哈希函数
    哈希函数公式：h(v) = floor((v · x + b) / w)
    其中v·x是向量点积，模拟随机超平面；b是偏移量；w是桶宽参数
    
    :param v: 输入的数据点向量，待计算哈希值的样本
    :param x: 随机生成的超平面法向量，维度与v一致，控制哈希函数的方向
    :param b: 随机偏移量，范围在[0, w)，用于增加哈希函数的随机性
    :param w: 桶宽参数，控制哈希桶的粒度，影响相似性判断的灵敏度
    :return: 计算得到的整数哈希值，相同哈希值表示数据点可能落在同一个哈希桶中
    """
    # 向量点积(v · x) + 偏移量b，再除以桶宽w后向下取整得到哈希值
    return (v.dot(x) + b) // w


def dealOneBuket(dataSet):
    """
    使用一个随机生成的哈希函数对整个数据集进行映射，将数据分配到哈希桶中
    每个哈希函数对应一组(x, b)参数，通过该函数将所有数据点映射到不同的桶（用哈希值表示）
    
    :param dataSet: 输入的数据集，numpy数组格式，shape为[样本数, 特征维度]
    :return: 列表，包含每个数据点对应的哈希值，相同值表示属于同一个哈希桶
    """
    k = dataSet.shape[1]  # 获取数据的特征维度k
    b = random.uniform(0, w)  # 生成[0, w)范围内的随机偏移量b
    x = np.random.random(k)  # 生成k维随机向量x（服从均匀分布），作为超平面法向量
    
    buket = []  # 存储所有数据点的哈希值
    for data in dataSet:  # 遍历每个数据点
        h = getHash(data, x, b, w)  # 计算当前数据点的哈希值
        buket.append(h)  # 将哈希值加入列表
    return buket


if __name__ == "__main__":
    # 定义示例数据集，包含7个样本，每个样本有6个特征
    # 从数据直观上看：前2个样本较相似，中间2个样本较相似，后3个样本较相似
    dataSet = [
        [8, 7, 6, 4, 8, 9],
        [7, 8, 5, 8, 9, 7],
        [3, 2, 0, 1, 2, 3],
        [3, 3, 2, 3, 3, 3],
        [21, 21, 22, 99, 2, 12],
        [1, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 0]
    ]
    dataSet = np.array(dataSet)  # 转换为numpy数组便于向量运算

    w = 4  # 哈希桶宽度参数，控制哈希粒度
    hash_funcs_num = 4  # 定义要生成的哈希函数数量

    # 循环生成多个哈希函数，分别对数据集进行映射并打印结果
    # 观察结果可发现：相似的样本在多数哈希函数下会映射到相同的哈希值
    for _ in range(hash_funcs_num):
        print(dealOneBuket(dataSet))