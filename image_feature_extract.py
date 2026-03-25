# -*- coding:utf-8 -*-  # 指定文件编码为UTF-8
"""  # 开始文件文档字符串
@file name  : image_feature_extract.py  # 文件名
@author     : TingsongYu https://github.com/TingsongYu  # 作者信息
@date       : 2023-04-30  # 创建日期
@brief      : 运行耗时约2-4h， 将coco数据集采用CLIP模型进行encoder，获得Nx512的特征矩阵，以及id到path的映射字典  # 功能说明
"""  # 结束文件文档字符串
import torch  # 导入PyTorch深度学习框架
import cv2  # 导入OpenCV计算机视觉库
import clip  # 导入CLIP多模态模型库
from PIL import Image  # 导入Python图像处理库
import pickle  # 导入Python序列化模块
from tqdm import tqdm  # 导入进度条显示模块

from config.base_config import CFG  # 导入基础配置参数
from my_utils.utils import get_file_path  # 导入工具函数获取文件路径
device = CFG.device  # 使用配置的设备（优先GPU，兼容CPU）


def main():  # 定义主函数
    clip_model, preprocess = clip.load(CFG.clip_backbone_type, device=device, jit=False)  # 加载CLIP模型和预处理器

    feat_list = []  # 初始化特征列表

    # 1. 获取图片路径  # 步骤1注释
    path_img_list = get_file_path(CFG.image_file_dir, ['jpg', 'JPEG'])  # 获取指定目录下所有jpg和JPEG格式的图片路径

    # 2. 推理  # 步骤2注释
    for path_img in tqdm(path_img_list):  # 遍历图片路径列表，显示进度条
        image_bgr = cv2.imread(path_img)  # 使用OpenCV读取图片，默认为BGR格式
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # 将BGR格式转换为RGB格式
        image = preprocess(Image.fromarray(image_rgb)).unsqueeze(0).to(device)  # 预处理图像并添加批次维度，移至设备
        with torch.no_grad():  # 禁用梯度计算以节省内存和计算资源
            img_feat_vec = clip_model.encode_image(image)  # 使用CLIP模型编码图像获得特征向量
            img_feat_vec = img_feat_vec / img_feat_vec.norm(dim=1, keepdim=True)  # 对特征向量进行L2归一化，缩放为单位向量
            img_feat_vec = img_feat_vec.cpu()  # 将特征向量移至CPU内存

        feat_list.append(img_feat_vec)  # 将特征向量添加到特征列表中

    # 3. 存储结果  # 步骤3注释
    feat_mat = torch.concat(feat_list)  # 将特征列表拼接为特征矩阵
    feat_mat = feat_mat.numpy()  # 将PyTorch张量转换为NumPy数组

    index_ = range(len(path_img_list))  # 创建索引序列，从0到图片数量-1
    map_dict = dict(zip(index_, path_img_list))  # 创建索引到图片路径的映射字典

    with open(CFG.feat_mat_path, 'wb') as f:  # 以二进制写入模式打开特征矩阵文件
        pickle.dump(feat_mat, f)  # 序列化并保存特征矩阵到文件

    with open(CFG.map_dict_path, 'wb') as f:  # 以二进制写入模式打开映射字典文件
        pickle.dump(map_dict, f)  # 序列化并保存映射字典到文件


if __name__ == '__main__':  # 当脚本作为主程序运行时

    filename = '000000000081.jpg'  # 测试文件名（未使用）
    main()  # 调用主函数
