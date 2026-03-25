# -*- coding:utf-8 -*-  # 指定文件编码为UTF-8
"""  # 开始文件文档字符串
@file name  : 01_retrieval_by_faiss.py  # 文件名
@author     : TingsongYu https://github.com/TingsongYu  # 作者信息
@date       : 2023-04-30  # 创建日期
@brief      : 基于faiss构建检索功能模块，包含IndexModule； CLIPModel； ImageRetrievalModule  # 功能说明
"""  # 结束文件文档字符串
import os  # 导入操作系统相关模块
import faiss  # 导入Facebook AI相似性搜索库
import matplotlib.pyplot as plt  # 导入Matplotlib绘图库
import torch  # 导入PyTorch深度学习框架
import cv2  # 导入OpenCV计算机视觉库
import clip  # 导入CLIP多模态模型库
from PIL import Image  # 导入Python图像处理库
import pickle  # 导入Python序列化模块
import time  # 导入时间相关模块
import numpy as np  # 导入NumPy数值计算库

from config.base_config import CFG  # 导入基础配置参数
from my_utils.utils import cv_imread  # 导入自定义图像读取工具函数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 根据CUDA可用性设置设备

# 检测是否支持GPU，仅在GPU模式下初始化GPU资源
try:
    res = faiss.StandardGpuResources()  # 初始化FAISS GPU资源对象
    gpu_id = 0  # 设置GPU ID为0
    USE_GPU = True
except AttributeError:
    # faiss-cpu不支持GPU资源
    res = None
    gpu_id = -1
    USE_GPU = False


class IndexModule(object):  # 定义索引器模块类
    """  # 类文档字符串开始
    索引器，将faiss功能封装到该类中，对外提供基于向量的检索功能
    """  # 类文档字符串结束
    def __init__(self, index_string, feat_dim, feat_mat):  # 构造函数
        self.index_string = index_string  # 存储索引类型字符串
        self.feat_dim = feat_dim  # 存储特征维度
        self.index = None  # 初始化索引对象为None
        self._init_index(feat_mat)  # 调用私有方法初始化索引

    def _init_index(self, feat_mat):  # 定义私有索引初始化方法
        """  # 方法文档字符串开始
        初始化faiss索引器， 包括训练及数据添加
        :param feat_mat: ndarray，  shape is N x feat_dim, CLIP的feat_dim是512维， 需要是float32  # 参数说明
        :return:  # 返回值说明
        """  # 方法文档字符串结束
        assert len(feat_mat.shape) == 2, f'feat_mat must be 2 dim, but got {feat_mat.shape}'  # 断言特征矩阵为2维
        if feat_mat.dtype != np.float32:  # 检查数据类型是否为float32
            feat_mat = feat_mat.astype(np.float32)  # 转换数据类型为float32
            print(f'feat_mat dtype is not float32, is {feat_mat.dtype}. convert done!')  # 打印转换信息

        index = faiss.index_factory(self.feat_dim, self.index_string)  # 使用FAISS工厂方法创建索引
        if USE_GPU and res is not None:
            self.index = faiss.index_cpu_to_gpu(res, gpu_id, index)  # 将索引从CPU迁移到GPU
            print('[IndexModule] 使用GPU索引')
        else:
            self.index = index  # CPU模式直接使用CPU索引
            print('[IndexModule] 使用CPU索引')

        s1 = time.time()  # 记录训练开始时间
        self.index.train(feat_mat)  # 使用特征矩阵训练索引
        s2 = time.time()  # 记录训练结束时间

        self.index.add(feat_mat)  # 将特征矩阵添加到索引中

        log_info = f'training time:{s2-s1} s, train set:{feat_mat.shape}'  # 创建日志信息
        print(log_info)  # 打印训练信息

    def feat_retrieval(self, feat_query, topk):  # 定义特征检索方法
        """  # 方法文档字符串开始
        执行特征检索，这里未编写批量查询的代码
        :param feat_query: 1x512的矩阵  # 查询特征向量
        :param topk:  # 返回的最近邻数量
        :return: distance与ids分别是L2距离值， 结果图片的索引序号  # 返回值说明
        """  # 方法文档字符串结束
        assert feat_query.shape == (1, CFG.feat_dim), f'query vec must be 1x512, but got {feat_query.shape}'  # 断言查询向量形状正确
        assert isinstance(topk, int), f'topk should be int, but got {type(topk)}'  # 断言topk为整数类型

        distance, ids = self.index.search(feat_query, topk)  # noqa: E741  # 执行搜索获取距离和索引
        ids = ids.squeeze(0)  # 压缩索引数组去除批次维度
        distance = distance.squeeze(0)  # 压缩距离数组去除批次维度
        return distance, ids  # 返回距离和索引数组

    def add_vectors(self, vectors):
        """
        增量添加向量到索引

        Args:
            vectors: 特征向量矩阵，shape (N, feat_dim)

        Returns:
            新添加向量的ID列表（从当前总数开始递增）
        """
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # 获取当前索引中的向量数
        start_id = self.index.ntotal

        # 添加到索引
        self.index.add(vectors)

        # 返回新分配的ID
        new_ids = list(range(start_id, start_id + len(vectors)))
        return new_ids

    def get_total_count(self):
        """获取索引中当前向量总数"""
        return self.index.ntotal


class CLIPModel(object):  # 定义CLIP模型类
    """  # 类文档字符串开始
    特征提取器，将clip模型封装到这里，对外提供特征提取功能
    """  # 类文档字符串结束
    def __init__(self, clip_backbone_type, device):  # 构造函数
        self.device = device  # 存储设备信息
        self.clip_backbone_type = clip_backbone_type  # 存储CLIP模型类型
        self.model, self.preprocess = clip.load(clip_backbone_type, device=device, jit=False)  # 加载CLIP模型和预处理器

    def encode_image_by_path(self, path_img):  # 定义按路径编码图像的方法
        """  # 方法文档字符串开始
        对图像进行编码，接收的是图片路径
        :param path_img:  # 图像文件路径参数
        :return:  # 返回值说明
        """  # 方法文档字符串结束
        # read img  # 注释：读取图像
        image_bgr = cv_imread(path_img)  # 使用自定义函数读取BGR格式图像
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB格式
        image = self.preprocess(Image.fromarray(image_rgb)).unsqueeze(0).to(device)  # 预处理图像并添加批次维度
        with torch.no_grad():  # 禁用梯度计算
            img_feat_vec = self.model.encode_image(image)  # 使用CLIP模型编码图像
            img_feat_vec /= img_feat_vec.norm(dim=-1, keepdim=True)  # 对特征向量进行L2归一化
            img_feat_vec = img_feat_vec.cpu().numpy()  # 1x512向量  # 转换为NumPy数组

        return img_feat_vec  # 返回图像特征向量

    def encode_image_by_ndarray(self, image_rgb):  # 定义按数组编码图像的方法
        """  # 方法文档字符串开始
        对图像进行编码，接收的是图像数组
        :param img_rgb:  # RGB图像数组参数
        :return:  # 返回值说明
        """  # 方法文档字符串结束
        assert image_rgb.ndim == 3, 'image_rgb 必须要是3d-array，但传入的是:{}维的！！'.format(image_rgb.ndim)  # 断言图像为3维数组
        # read img  # 注释：处理图像
        image = self.preprocess(Image.fromarray(image_rgb)).unsqueeze(0).to(device)  # 预处理NumPy数组图像
        with torch.no_grad():  # 禁用梯度计算
            img_feat_vec = self.model.encode_image(image)  # 使用CLIP模型编码图像
            # # 一定要Normalization！https://github.com/rom1504/clip-retrieval/blob/main/clip_retrieval/clip_back.py#L226  # 归一化重要性注释
            img_feat_vec /= img_feat_vec.norm(dim=-1, keepdim=True)  # 对特征向量进行L2归一化
            img_feat_vec = img_feat_vec.cpu().numpy()  # 1x512向量  # 转换为NumPy数组

        return img_feat_vec  # 返回图像特征向量

    def encode_text_by_string(self, text):  # 定义按字符串编码文本的方法
        """  # 方法文档字符串开始
        对图像进行编码，接收的是文本描述
        :param text:  # 文本描述参数
        :return:  # 返回值说明
        """  # 方法文档字符串结束
        token = clip.tokenize([text]).to(self.device)  # 对文本进行分词并移至设备
        feat_text = self.model.encode_text(token)  # 使用CLIP模型编码文本
        # feat_text /= feat_text.norm(dim=-1, keepdim=True)  # !!! 图片需要Normalization，text不需要  # 文本不需要归一化的注释
        feat_text = feat_text.detach().cpu().numpy()  # 1x512向量  # 分离张量并转换为NumPy数组

        return feat_text  # 返回文本特征向量


class ImageRetrievalModule(object):  # 定义图像检索模块类
    """  # 类文档字符串开始
    图像检索模块，组装特征提取器、索引器，对外实现query的特征提取、索引
    """  # 类文档字符串结束
    def __init__(self, index_string, feat_dim, feat_mat, map_dict, backbone, device):  # 构造函数
        self.index_model = IndexModule(index_string, feat_dim, feat_mat)  # 初始化索引模型
        self.clip_model = CLIPModel(backbone, device)  # 初始化CLIP模型
        self.map_dict = map_dict  # 存储索引到路径的映射字典

    def retrieval_func(self, query_info, topk):  # 定义检索功能方法
        """  # 方法文档字符串开始
        根据查询信息进行图像检索，可以输入text或者图片路径
        :param query_info:  # 查询信息，可以是文本或图像路径
        :param topk:  # 返回结果数量
        :return:  # 返回值说明
        """  # 方法文档字符串结束
        if os.path.exists(query_info):  # 如果查询信息是存在的文件路径
            feat_vec = self.clip_model.encode_image_by_path(query_info)  # 使用图像路径编码
        elif isinstance(query_info, str):  # 如果查询信息是字符串
            feat_vec = self.clip_model.encode_text_by_string(query_info)  # 使用文本编码
        elif isinstance(query_info, np.ndarray):  # 如果查询信息是NumPy数组
            feat_vec = self.clip_model.encode_image_by_ndarray(query_info)  # 使用图像数组编码

        feat_vec = feat_vec.astype(np.float32)  # 转换特征向量为float32类型
        distance_, id_ = self.index_model.feat_retrieval(feat_vec, topk)  # 执行特征检索

        result_path_list = [self.map_dict.get(id_tmp, 'None') for id_tmp in id_]  # 根据索引获取图像路径列表
        return distance_, id_, result_path_list  # 返回距离、索引和路径列表

    def visual_result(self, query_info, distances, ids, path_list):  # 定义结果可视化方法
        """  # 方法文档字符串开始
        绘制检索结果图像
        :param query_info: str, 图片路径或者text  # 查询信息参数
        :param distances:  # 距离数组参数
        :param ids:  # 索引数组参数
        :param path_list:  # 路径列表参数
        :return: None  # 无返回值
        """  # 方法文档字符串结束
        plt.figure(figsize=(12, 12))  # 创建图形窗口，设置大小为12x12
        subplot_num = int(np.floor(np.sqrt(len(ids))) + 1)  # 计算子图网格大小

        if os.path.exists(query_info):  # 如果查询信息是文件路径
            img_ = Image.open(query_info)  # 打开查询图像
            plt.subplot(subplot_num, subplot_num, np.square(subplot_num))  # 创建子图
            plt.imshow(img_)  # 显示查询图像
            plt.title('query image')  # 设置子图标题
        else:  # 如果查询信息是文本
            plt.subplot(subplot_num, subplot_num, np.square(subplot_num))  # 创建子图
            plt.text(.1, .1, query_info, fontsize=12)  # 显示查询文本

        for ii, (distance, id, path_) in enumerate(zip(distances, ids, path_list)):  # 遍历检索结果
            if id == -1:  # 如果索引无效
                continue  # 跳过当前循环

            img_ = Image.open(path_)  # 打开结果图像
            plt.subplot(subplot_num, subplot_num, ii + 1)  # 创建子图
            plt.imshow(img_)  # 显示结果图像
            plt.title(str(distance))  # 设置子图标题为距离值

        plt.subplots_adjust(wspace=0.5)  # 调整子图水平间距
        plt.subplots_adjust(hspace=0.5)  # 调整子图垂直间距
        plt.show()  # 显示图形


def main(query):  # 定义主函数
    """  # 函数文档字符串开始
    测试图像检索功能。
    前提：image_feature_extract.py已经执行完毕，在data/文件夹下有对应的pkl文件
    :param query: string， text or path_img  # 查询参数
    :return:  # 返回值说明
    """  # 函数文档字符串结束

    with open(CFG.feat_mat_path, 'rb') as f:  # 以二进制读取模式打开特征矩阵文件
        feat_mat = pickle.load(f)  # 加载特征矩阵
    with open(CFG.map_dict_path, 'rb') as f:  # 以二进制读取模式打开映射字典文件
        map_dict = pickle.load(f)  # 加载映射字典

    # 初始化图像检索模块  # 注释
    ir_model = ImageRetrievalModule(CFG.index_string, CFG.feat_dim, feat_mat, map_dict,
                                    CFG.clip_backbone_type, CFG.device)  # 创建图像检索模块实例

    # 调用图像检索功能  # 注释
    distance_result, index_result, path_list = ir_model.retrieval_func(query, CFG.topk)  # 执行检索

    # 可视化结果  # 注释
    ir_model.visual_result(query, distance_result, index_result, path_list)  # 可视化检索结果


if __name__ == '__main__':  # 当脚本作为主程序运行时
    # 设置query，可选文本，或者是图像  # 注释

    path_img = '000000000154.jpg'  # 测试图像路径
    query = path_img  # 设置查询为图像路径
    query = 'an image of a {}'.format('car')  # zebra  # 设置查询为文本描述

    main(query)  # 调用主函数
