# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import sys
import time
import numpy as np
import os

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

dir_path = r'E:\py_code\deep-learning-data\sift_1m'
def load_sift1M():
    print("Loading sift1M...", end='', file=sys.stderr)
    
    # 拼接目录与文件名
    xt_path = os.path.join(dir_path, "sift_learn.fvecs")
    xb_path = os.path.join(dir_path, "sift_base.fvecs")
    xq_path = os.path.join(dir_path, "sift_query.fvecs")
    gt_path = os.path.join(dir_path, "sift_groundtruth.ivecs")
    
    # 读取文件（假设fvecs_read和ivecs_read已定义）
    xt = fvecs_read(xt_path)
    xb = fvecs_read(xb_path)
    xq = fvecs_read(xq_path)
    gt = ivecs_read(gt_path)
    
    print("done", file=sys.stderr)
    return xb, xq, xt, gt


def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    D, I = index.search(xq, k)  # noqa: E741
    t1 = time.time()

    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)   # nok = (I[:, :rank] == gtc).sum()
        i *= 10

    return (t1 - t0) * 1000.0 / nq, recalls
