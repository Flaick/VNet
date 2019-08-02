"""
训练脚本
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import glob
from time import time
import torch.nn as nn
import SimpleITK as sitk
import xlsxwriter as xw
from common.Metric import Metric
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset.my_dataset import CTDataLoader
from net.ResUnet import ResUNet, init, StageNet
from common.Visualizer import Visualizer
import torch.nn.functional as F


cudnn.benchmark = True
checkpoint_dir = '/data/qinwang/multistage_yk/' #0.646
prediction_save_dir = '/data/qinwang/multistage_yk/KiTS/'


### For valid set  ### 
# prediction_save_dir = '/data/weihao/pre-KiTS-3mm/predictions_yk'
#os.mkdir(prediction_save_dir)
test_root = '/data/weihao/pre-KiTS-3mm/val/CT/'
test_GT = '/data/weihao/pre-KiTS-3mm/val/GT/'

test_list = glob.glob(test_root+'*.nii')
test_list = [file for file in test_list if 'back' not in file]

# define the model
# net = HighRes3DNet(inchannel=1, num_organ=1)
net = StageNet(training=False)
net = torch.nn.DataParallel(net).cuda()

# load the weight
checkpoint_path = checkpoint_dir+'epoch_67/net_epoch_67.pth'
net.load_state_dict(torch.load(checkpoint_path))
net.eval()
print('The model load weight completed!')

metricer = Metric()

start = time()

mean_loss = []
train_image = None
train_label = None
train_prediciton = None
train_label_path = None
mean=[]
slice_number = 32 #slice numbers
interval = 4

with torch.no_grad():
    for step, image_path in enumerate(test_list):
        image_org = sitk.ReadImage(image_path, sitk.sitkInt16)
        image_array = sitk.GetArrayFromImage(image_org)  # [384, 240, 80] ndarray in range [-250, 250]
        image_array = image_array / 250.0
        glob_cube = torch.FloatTensor(image_array)
        glob_cube = glob_cube.permute((2, 0, 1))
        glob_cube = glob_cube.unsqueeze(dim=0)
        glob_cube = glob_cube.unsqueeze(dim=0)


        # 处理完毕，将array转换为tensor
        start_slice = 0
        end_slice = start_slice + slice_number - 1 #31
        ct_array_list = [] 

        while end_slice <= image_array.shape[2] - 1:
            ct_array_list.append(image_array[:,:,start_slice:end_slice + 1])
            start_slice = start_slice + interval #4
            end_slice = start_slice + slice_number - 1 #47

        # 当无法整除的时候反向取最后一个block
        if end_slice is not image_array.shape[2] - 1 + slice_number:
            count = image_array.shape[2] - start_slice #count is slice number of last cube
            ct_array_list.append(image_array[:,:,-slice_number:])
        # print('outputs_list',outputs_list)
        start_slice1 = 0
        end_slice1 = start_slice1  + slice_number - 1 #31
        res_numpy = np.zeros((3,image_array.shape[-1],384,240))#,ct_array.shape[-1],ct_array.shape[0],ct_array.shape[1]))
        ind_1 = 0
        while end_slice1 <= image_array.shape[-1] - 1:
            ct_tensor = torch.FloatTensor(ct_array_list[ind_1]).cuda()
            ct_tensor = ct_tensor.permute((2, 0, 1))
            ct_tensor = ct_tensor.unsqueeze(dim=0)
            ct_tensor = ct_tensor.unsqueeze(dim=0)
            outputs1, outputs2 = net(ct_tensor, glob_cube, start_slice1, end_slice1)  # after softmax
            outputs1 = outputs1.squeeze()
            outputs2 = outputs2.squeeze()
            res_map1 = F.softmax(outputs1, dim=1).cpu().numpy()
            res_map2 = F.softmax(outputs2, dim=1).cpu().numpy()


            res_numpy[:,start_slice1:end_slice1 + 1,:,:] =   res_numpy[:,start_slice1:end_slice1 + 1,:,:]+res_map1[:,start_slice1:end_slice1 + 1,:,:]#res_numpy[:,start_slice1:end_slice1 + 1,:,:]#res_map2#res_map1[:,start_slice1:end_slice1 + 1,:,:] +
            start_slice1 = start_slice1 + interval #
            end_slice1 = start_slice1 + slice_number - 1
            ind_1 = ind_1 + 1
            #print(start_slice1,end_slice1,ind)


        # 当无法整除的时候反向取最后一个block
        if end_slice1 is not res_numpy.shape[2] - 1 + slice_number:
            ct_tensor = torch.FloatTensor(ct_array_list[-1]).cuda()
            ct_tensor = ct_tensor.permute((2, 0, 1))
            ct_tensor = ct_tensor.unsqueeze(dim=0)
            ct_tensor = ct_tensor.unsqueeze(dim=0)


            outputs1, outputs2 = net(ct_tensor,glob_cube, res_numpy.shape[1] - slice_number, res_numpy.shape[1])  # after softmax
            outputs1 = outputs1.squeeze()
            outputs2 = outputs2.squeeze()
            res_map1 = F.softmax(outputs1, dim=1).cpu().numpy()
            res_map2 = F.softmax(outputs2, dim=1).cpu().numpy()
            res_numpy[:,-slice_number:,:,:] = res_map1[:,-slice_number:,:,:] + res_numpy[:,-slice_number:,:,:]#res_map2#res_map1[:,-slice_number:,:,:]


        res_map = np.argmax(res_numpy,axis=0)
        res_map = res_map.transpose((1,2,0))
        prediction = res_map
        prediction = prediction.astype(np.int8)
        pred_seg = sitk.GetImageFromArray(prediction)
        pred_seg.SetDirection(image_org.GetDirection())#image_org
        pred_seg.SetOrigin(image_org.GetOrigin())
        pred_seg.SetSpacing(image_org.GetSpacing())
        fpath, fname = os.path.split(image_path)
        sitk.WriteImage(pred_seg, os.path.join(prediction_save_dir, fname.replace('img', 'pred')))
        print(image_path+' completed!')