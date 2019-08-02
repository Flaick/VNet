import os
import shutil
from time import time

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
import math
import random
from torch.autograd import Variable
import torch
import torch.nn.functional as F

# 新产生的训练数据存储路径
# if os.path.exists('/data/weihao/KiTS-test-ROI/%s'%dataset) is True:
#     shutil.rmtree('/data/weihao/KiTS-test-ROI/%s'%dataset)
# os.mkdir('/data/weihao/KiTS-test-ROI/%s'%dataset)
# os.mkdir(new_ct_path)

start_time = time()
f = open('/home/qinwang/ZWB-KiTS19-Base/yuankun/test_bbox_yk_single.txt')

for line in f.readlines():  # 依次读取每行
    ori_line = line.strip().split(',')  # 去掉每行头尾空白
    # print("读取的数据为: %s" % (line))
    ct_path = ori_line[0]
    file = "/data/weihao/KiTS2019-3mm-test-clip/%s.nii" % ct_path.split('.')[0]
    bbox = np.array(list(map(int, ori_line[1:])))

    ct = sitk.ReadImage(file, sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    x = (bbox[0] + bbox[3])//2
    y = (bbox[1] + bbox[4])//2
    z = (bbox[2] + bbox[5])//2
    m_x, m_y, m_z = x-192, y-120, z-40
    w_x, w_y, w_z = x+192, y+120, z+40
    if m_x < 0:
        w_x += -m_x
        m_x = 0
    if m_y < 0:
        w_y += -m_y
        m_y = 0
    if m_z < 0:
        w_z += -m_z
        m_z = 0
    new_ct_array = ct_array[m_x:w_x, m_y:w_y, m_z:w_z]

    if w_z > ct_array.shape[-1]:
        pad = w_z - ct_array.shape[-1]
        new_ct_array = np.pad(new_ct_array, ((0,0),(0,0),(pad, 0)), constant_values=0, mode='constant')

    if w_y > ct_array.shape[-2]:
        pad = w_y - ct_array.shape[-2]
        new_ct_array = np.pad(new_ct_array, ((0,0),(pad, 0),(0,0)), constant_values=0, mode='constant')

    if w_x > ct_array.shape[-3]:
        pad = w_x - ct_array.shape[-3]
        new_ct_array = np.pad(new_ct_array, ((pad, 0),(0,0),(0,0)), constant_values=0, mode='constant')

    print(new_ct_array.shape)
    if new_ct_array.shape != (384,240,80):
        print('fuck')
        exit(0)

    # 保存数据
    new_ct = sitk.GetImageFromArray(new_ct_array)
    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing((ct.GetSpacing()))
    index = ct_path.split('.')[0].split('_')[-1]
    new_ct_name = 'img-' + index + '.nii'

    sitk.WriteImage(new_ct, os.path.join("/data/weihao/testset2/", new_ct_name))