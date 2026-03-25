# -*- coding:utf-8 -*-
"""
@file name  : 02_pq_benchmark.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-04-29
@brief      : 基于faiss库测试IVF（倒排文件）+ Product Quantization（乘积量化）算法的性能。
              主要通过调整IVF中的聚类中心数量和probe参数，分析其对检索耗时和召回率（recall）的影响。
              测试数据采用SIFT1M数据集，需从http://corpus-texmex.irisa.fr/下载sift.tar.gz(161MB)
@analysis   :耗时和召回率都随 probe 增大而增加，耗时随聚类中心数量增大而减少，召回率先升后稳，整体略有波动
            当聚类中心数量为 4096 时，训练数据量（10 万）不足，可能导致聚类中心代表性差，影响召回率
"""
from __future__ import print_function  # 兼容Python2的print函数语法
import faiss  # FAISS库，用于实现高效向量检索
import pickle  # 用于序列化/反序列化实验结果，便于存储和读取
from faiss_datasets import load_sift1M, evaluate  # 自定义数据集加载和评估函数（需提前准备）
from matplotlib import pyplot as plt  # 用于绘制性能曲线（召回率、耗时）


def main():
    """主函数：加载数据、构建索引、执行检索并记录性能指标"""

    # 1 加载SIFT1M数据集
    # 数据集说明：SIFT1M包含100万条基础向量、1万条查询向量、训练向量及对应的ground truth（真实近邻标签）
    # 数据需提前下载并解压，文件夹命名为sift1M且与当前脚本同目录
    xb, xq, xt, gt = load_sift1M()  # xb:基础库向量, xq:查询向量, xt:训练向量, gt:真实近邻索引
    nq, d = xq.shape  # nq:查询向量数量, d:向量维度（SIFT特征通常为128维）

    # 2 初始化实验参数与结果存储
    results_dict = {}  # 用于存储不同参数组合下的性能结果（键：参数组合，值：[耗时, r1, r10, r100]）
    
    # 注意：PQ算法中除8bit量化外，GPU加速支持有限，因此本实验使用CPU运行
    # res = faiss.StandardGpuResources()  # 初始化GPU资源（此处注释表示不使用GPU）
    # flat_config = faiss.GpuIndexFlatConfig()  # GPU配置（未使用）
    # flat_config.device = 0  # 指定使用第0块GPU（未使用）

    # 遍历不同的聚类中心数量（IVF的核心参数之一）
    for center_num in center_nums:
        # 定义索引名称：IVF{center_num},PQ32x8表示聚类中心数为center_num，PQ分为32个子向量，每个子向量用8bit量化
        pq_index_string = "IVF{},PQ32x8".format(center_num)
        # 创建索引：使用faiss的index_factory构建IVF+PQ索引，d为向量维度
        # PQ32x8参数说明：32表示将原向量拆分为32个子向量，8表示每个子向量用8bit量化（即每个子向量映射到2^8=256个聚类中心）
        index = faiss.index_factory(d, pq_index_string)
        
        # 若使用GPU，可通过以下代码将索引迁移至GPU（本实验禁用）
        # index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # 训练索引：使用训练集xt对索引进行训练（主要是学习PQ的量化参数和IVF的聚类中心）
        index.train(xt)
        # 添加基础库向量到索引：将xb存入索引，构建检索结构
        index.add(xb)
        # 预热操作：首次检索可能包含初始化开销，先执行一次无记录的检索消除影响
        index.search(xq, 123)

        # 遍历不同的probe参数（IVF的核心参数之二，控制检索时访问的聚类中心数量）
        for probe in probes:
            index.nprobe = probe  # 设置当前检索的probe值
            # 评估检索性能：返回平均耗时t（毫秒）和召回率r（r[1]、r[10]、r[100]分别表示top1、top10、top100的召回率）
            t, r = evaluate(index, xq, gt, 100)  # 100表示评估时参考top100的真实近邻
            # 构建结果键名，格式为"IVF{center_num},PQ32x8_probe_{probe}"
            results_key = '{}_probe_{}'.format(pq_index_string, probe)
            # 存储结果：[耗时, top1召回率, top10召回率, top100召回率]
            results_dict[results_key] = [t, r[1], r[10], r[100]]
            # 打印当前参数组合的性能结果
            print("{}: {:.3f} ms, recalls = {:.4f}, {:.4f}, {:.4f}".format(
                results_key, t, r[1], r[10], r[100]))

    # 将实验结果序列化并保存到本地文件，便于后续绘图分析
    with open(path_pkl, 'wb') as f:
        pickle.dump(results_dict, f)


def plot_curve():
    """绘制性能曲线：展示不同参数下的召回率和耗时变化趋势"""
    # 从本地文件加载实验结果
    with open(path_pkl, 'rb') as f:
        results_dict = pickle.load(f)

    # 绘制不同聚类中心数量下，召回率随probe的变化曲线
    # 子图布局：3行2列（对应center_nums的6个元素）
    for idx, center_num in enumerate(center_nums):
        # 初始化当前聚类中心数量下的性能列表
        time_list_sub, r1_l, r10_l, r100_l = [], [], [], []
        # 遍历所有probe值，提取对应结果
        for probe in probes:
            pq_index_string = "IVF{},PQ32x8_probe_{}".format(center_num, probe)
            t, r1, r10, r100 = results_dict[pq_index_string]
            time_list_sub.append(t)
            r1_l.append(r1)       # top1召回率
            r10_l.append(r10)     # top10召回率
            r100_l.append(r100)   # top100召回率

        # 绘制子图
        x = range(len(probes))  # x轴为probe的索引
        plt.subplot(3, 2, idx+1)  # 定位到第idx+1个子图
        plt.plot(x, r1_l, label='r1')    # 绘制top1召回率曲线
        plt.plot(x, r10_l, label='r10')  # 绘制top10召回率曲线
        plt.plot(x, r100_l, label='r100')# 绘制top100召回率曲线
        plt.legend()  # 显示图例
        plt.xticks(x, probes)  # x轴刻度设为probe值
        plt.xlabel('nprobe')   # x轴标签
        plt.ylabel('recall')   # y轴标签
        plt.ylim(0, 1.2)       # y轴范围限制在0~1.2（便于对比）
        plt.title(f'{center_num} centroid')  # 子图标题（当前聚类中心数量）

    # 调整子图间距，避免重叠
    plt.subplots_adjust(wspace=0.5)  # 水平间距
    plt.subplots_adjust(hspace=1)    # 垂直间距
    plt.suptitle(f'IVFxxxx,PQ32x8  center_nums: {center_nums}')  # 总标题
    plt.show()  # 显示召回率曲线

    # 绘制不同聚类中心数量下，耗时随probe的变化曲线
    for idx, center_num in enumerate(center_nums):
        time_list = []  # 存储当前聚类中心数量下的耗时
        for probe in probes:
            pq_index_string = "IVF{},PQ32x8_probe_{}".format(center_num, probe)
            t, _, _, _ = results_dict[pq_index_string]
            time_list.append(t)
        x = range(len(probes))
        plt.plot(x, time_list, label='IVF{},PQ32x8'.format(center_num))  # 绘制耗时曲线

    plt.legend()  # 显示图例
    plt.xticks(x, probes)  # x轴刻度设为probe值
    plt.xlabel('nprobe')   # x轴标签
    plt.ylabel('time pass (ms)')  # y轴标签（耗时，单位毫秒）
    plt.title(f'Time Comparison')  # 图表标题
    plt.show()  # 显示耗时曲线

    a = 1  # 占位符，无实际作用


if __name__ == '__main__':
    # 实验参数配置
    center_nums = [128, 256, 512, 1024, 2048, 4096]  # IVF的聚类中心数量列表（需为2的幂，FAISS推荐）
    # center_nums = [1024, 2048, 3072, 4096]  # 备选聚类中心数量（注释掉表示不使用）
    probes = [1, 2, 4, 8, 16, 32, 64]  # IVF的probe参数列表（检索时访问的聚类中心数量）

    # 实验结果存储路径：格式化字符串，包含IVF和PQ参数
    path_pkl = 'results_dict_ivf{}pq16x8.pkl'

    # 执行主函数（运行实验并保存结果）
    main()
    # 绘制性能曲线（需先运行main()生成结果文件，再取消注释执行）
    # plot_curve()