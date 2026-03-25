# -*- coding:utf-8 -*-
"""
@file name  : 00_lsh_demo.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-04-21
@brief      : faiss库安装后的初步调用示例，展示了使用FAISS进行向量检索的基本流程，
              包括数据构建、索引创建、向量检索以及GPU加速等核心功能，帮助理解FAISS的基础用法
"""
import time  # 用于计算程序运行时间，评估检索效率
import numpy as np  # 用于生成和处理向量数据
import faiss  # FAISS库，Facebook开发的高效向量检索库，支持多种索引方式和加速策略


# ============================ step 0: 数据构建 ============================
# 设置随机种子，保证实验结果可复现
np.random.seed(1234)

d = 64                           # 向量维度，这里定义为64维向量
nb = 100000                      # 数据库中向量的数量，即待检索的样本总量
nq = 10000                       # 查询向量的数量，即需要查找相似向量的样本数

# 生成数据库向量：随机生成(nb, d)形状的数组，数据类型为float32（FAISS要求输入为32位浮点数）
xb = np.random.random((nb, d)).astype('float32')
# 生成查询向量：随机生成(nq, d)形状的数组，同样为float32类型
xq = np.random.random((nq, d)).astype('float32')

# 为向量添加可区分的趋势，使检索结果更具规律性（便于验证检索正确性）
# 对数据库向量的第0维进行偏移：按索引递增添加偏移量，使向量间存在可预测的差异
xb[:, 0] += np.arange(nb) / 1000.
# 对查询向量的第0维进行类似偏移，与数据库向量形成对应关系
xq[:, 0] += np.arange(nq) / 1000.


# ============================ step 1: 构建索引器 ============================
# 创建FlatL2索引：这是一种精确检索索引，基于L2（欧氏距离）度量，不进行任何近似
# 优点是检索结果精确，缺点是当数据量极大时速度较慢
index = faiss.IndexFlatL2(d)

# 将数据库向量添加到索引中：索引会对向量进行组织（FlatL2为暴力搜索，此处主要是存储向量）
index.add(xb)


# ============================ step 2: 索引（向量检索） ============================
k = 4  # 每个查询向量需要返回的最相似向量的数量（top-k检索）

# 多次执行检索并计时，观察平均性能
for i in range(5):
    s = time.time()  # 记录开始时间
    # 执行检索：xq为查询向量，k为返回的近邻数
    # 返回结果：D是距离矩阵（shape为[nq, k]，存储每个查询向量与对应近邻的距离）
    #           I是索引矩阵（shape为[nq, k]，存储每个查询向量的近邻在数据库中的索引）
    D, I = index.search(xq, k)
    # 计算并打印检索耗时
    print("{}*{}量级的精确检索，耗时:{:.3f}s".format(nb, nq, time.time()-s))


# ============================ step 3: 检查索引结果 ============================
# 打印距离矩阵的形状和第一个查询向量的距离结果
print('D.shape: {}, D[0, ...]: {}'.format(D.shape, D[0]))
# 打印索引矩阵的形状和第一个查询向量的近邻索引
print('I.shape: {}, I[0, ...]: {}'.format(I.shape, I[0]))
# 结果说明：
# D中存储的是查询向量与每个近邻向量的L2距离，值越小表示越相似
# I中存储的是近邻向量在数据库xb中的索引（范围0~nb-1），可通过索引从xb中获取对应向量


# ============================ step 4: GPU 加速检索 ============================
# 1. 初始化GPU资源：创建GPU资源对象，管理GPU内存和计算资源
res = faiss.StandardGpuResources()

# 2. 先创建CPU版本的FlatL2索引（FAISS的GPU索引通常基于CPU索引转换）
index_flat = faiss.IndexFlatL2(d)

# 3. 将CPU索引迁移到GPU：参数0表示使用第0块GPU（多GPU时可指定其他编号）
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

# 将数据库向量添加到GPU索引中（数据会被传输到GPU内存）
gpu_index_flat.add(xb)

k = 4  # 保持与CPU检索相同的top-k参数

# 测试GPU加速的检索性能，同样执行5次
for i in range(5):
    s = time.time()  # 记录开始时间
    # 调用GPU索引进行检索，接口与CPU版本一致
    D, I = gpu_index_flat.search(xq, k)
    # 打印GPU检索耗时（通常会比CPU快很多，尤其数据量大时）
    print("{}*{}量级的精确检索，耗时:{:.3f}s".format(nb, nq, time.time()-s))

# 打印GPU检索的结果形状和第一个查询的结果，验证与CPU检索结果一致性（理论上应完全相同）
print('D.shape: {}, D[0, ...]: {}'.format(D.shape, D[0]))
print('I.shape: {}, I[0, ...]: {}'.format(I.shape, I[0]))