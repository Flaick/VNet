import os
import numpy as np
import torch.nn.functional as F
import glob
from plotly import tools
import random
import torch
import scipy.ndimage as ndimage
import SimpleITK as sitk
from torch.utils.data import Dataset
import plotly
import plotly.graph_objs as go
import config
lower = -250
class CTDataLoader(Dataset):
    def __init__(self, root = '/data/weihao/pre-KiTS-3mm/',mode = 'train', slice_number = None, scale=False, rotate=False, flip = False, glob_flag=False, use_weight=False):
        '''
        :param root:
        :param mode:
        '''
        super(CTDataLoader, self).__init__()
        self.root = root
        self.mode = mode
        self.scale = scale # the True of False
        self.slice_number = slice_number
        self.rotate = rotate # the True of False
        self.flip = flip # the True of False
        # self.num_class = config.num_class # the class number which include the background class
        self.glob_flag = glob_flag
        if self.mode == 'train':
            # self.image_dir = os.path.join(self.root, 'train/CT/')
            # self.label_dir = os.path.join(self.root, 'train/GT/')
            self.image_dir = os.path.join(self.root, 'train_original.txt') # read the case list by a txt file
            self.hard_image_dir = os.path.join(self.root, 'train_selected.txt')

            hf = open(self.hard_image_dir, 'r')
            self.image_hard_path_list = hf.readlines()
            self.image_hard_path_list = [file[:-1] for file in self.image_hard_path_list]
            self.image_hard_path_list.sort()

        elif self.mode == 'val':
            # self.image_dir = os.path.join(self.root, 'val/CT/')
            # self.label_dir = os.path.join(self.root, 'val/GT/')
            self.image_dir = os.path.join(self.root, 'val_original.txt')  # read the case list by a txt file

        # self.image_path_list = glob.glob(self.image_dir+'*.nii')
        # self.image_path_list = [file for file in self.image_path_list if 'back' not in file]
        # self.image_path_list.sort()

        f = open(self.image_dir, 'r')
        self.image_path_list = f.readlines()
        self.image_path_list = [file[:-1] for file in self.image_path_list]
        self.image_path_list.sort()

        from tqdm import tqdm
        if use_weight:
            labelweights = np.zeros(3)
            for image_path in tqdm(self.image_path_list,total=len(self.image_path_list)):
                label_path = image_path.replace('CT', 'GT')
                label_path = label_path.replace('img', 'label')
                label = sitk.ReadImage(label_path, sitk.sitkUInt8)
                label_array = sitk.GetArrayFromImage(label)  # [384, 240, 80] ndarray in 0,1,2
                tmp, _ = np.histogram(label_array, range(4))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
            print(self.labelweights)
        else:
            self.labelweights = np.ones(3)

    def __getitem__(self, index):
        """
        :param index:
        :return: torch.Size([batch, 1, 48, 384, 240]) torch.Size([batch, 48, 384, 240])
        """
        # if self.mode == 'train' and random.uniform(0,1) <= 0.7:
        #     image_path = self.image_hard_path_list[index%50] #只取前50难的case
        # else:
        #     image_path = self.image_path_list[index]
        image_path = self.image_path_list[index]
        label_path = image_path.replace('CT','GT')
        label_path = label_path.replace('img','label')

        image = sitk.ReadImage(image_path, sitk.sitkInt16)
        label = sitk.ReadImage(label_path, sitk.sitkUInt8)

        label_array = sitk.GetArrayFromImage(label) # [384, 240, 80] ndarray in 0,1,2
        #label_array[label_array != 2] = 0 #Get the Tumor label

        image_array = sitk.GetArrayFromImage(image) # [384, 240, 80] ndarray in range [-250, 250]
        image_array = self.clip_intensity(image_array)

        cube_glob = image_array
        label_glob = label_array
        # if config.num_class == 2 and config.organ_type[0] == 'tumor':
        #     label_array[label_array < 2] = 0
        #     label_array[label_array > 0] = 1
        # elif config.num_class ==2 and config.organ_type[0] == 'kidney':
        #     label_array[label_array > 0] = 1
        #
        # assert len(label_array.shape) == 3,'the error in crop'

        # if self.num_class is not None: # check the correctness of label
        #     assert label_array.min() >= 0 and label_array.max() < self.num_class, \
        #         'the range of file {} should be [0,num_class-1], but min {}--max{}'.format(image_path, label_array.min(),label_array.max())
        _, _, _, _, z_min, z_max = self.getBoundbox(label_glob)
        if self.slice_number is not None:
            # 在slice平面内随机选取self.slice_number 张slice
            start_slice = random.randint(max(z_min - 16,0), min(max(z_max-8, z_min),label_glob.shape[2]-self.slice_number))#(0, image_array.shape[-1] -self.slice_number)
            end_slice = start_slice + self.slice_number - 1
            # print(start_slice, end_slice, label_glob.shape[2])
            if end_slice >= label_glob.shape[2]:
                print('no!!!!')
                exit()
            image_array = image_array[:,:, start_slice:end_slice + 1]
            label_array = label_array[:,:, start_slice:end_slice + 1]
        else:
            start_slice = 0
            end_slice = 0

        # 处理完毕，将array转换为tensor

        image_array = torch.FloatTensor(image_array).unsqueeze(0) # [1, 384, 240, 80]
        image_array = image_array / 250.0 # rescale the range of intensity
        image_array = image_array.permute(0,3,1,2) # [1, 80, 384, 240]
        label_array = torch.FloatTensor(label_array) # [384, 240, 80]
        label_array = label_array.permute(2,0,1) # [80, 384, 240] nn.CrossEntropyLoss()

        cube_glob = torch.FloatTensor(cube_glob).unsqueeze(0) # [1, 384, 240, 80]
        cube_glob = cube_glob / 250.0 # rescale the range of intensity
        cube_glob = cube_glob.permute(0,3,1,2) # [1, 80, 384, 240]
        label_glob = torch.FloatTensor(label_glob) # [384, 240, 80]
        label_glob = label_glob.permute(2,0,1) # [80, 384, 240] nn.CrossEntropyLoss()

        #assert label_array.max() == 2, 'label needs to have 2 labels'
        # if self.num_class is not None: # check the correctness of label
        #     assert label_array.min() >= 0 and label_array.max() < self.num_class, \
        #     'the range of file {} should be [0,num_class-1], but min {}--max{}'.format(image_path, label_array.min(),label_array.max())
        if self.glob_flag:
            return image_array, label_array, cube_glob, start_slice, end_slice, label_glob
        else:
            return image_array, label_array

    def __len__(self):
        return len(self.image_path_list)

    def clip_intensity(self, ct_array, intensity_range=(-250,250)):
        ct_array[ct_array>intensity_range[1]] = intensity_range[1]
        ct_array[ct_array<intensity_range[0]] = intensity_range[0]
        return ct_array

    def zoom(self, ct_array, seg_array, patch_size):

        shape = ct_array.shape # [384, 240, 80]
        length_hight = int(shape[0] * patch_size)
        length_width = int(shape[1] * patch_size)

        length = int(256 * patch_size)

        x1 = int(random.uniform(0, shape[0] - length_hight))
        y1 = int(random.uniform(0, shape[1] - length_width))

        x2 = x1 + length_hight
        y2 = y1 + length_width

        ct_array = ct_array[x1:x2 + 1, y1:y2 + 1,:]
        seg_array = seg_array[x1:x2 + 1, y1:y2 + 1,:]

        with torch.no_grad():

            ct_array = torch.FloatTensor(ct_array).unsqueeze(dim=0).unsqueeze(dim=0)
            ct_array = ct_array
            ct_array = F.interpolate(ct_array, (shape[0], shape[1], shape[2]), mode='trilinear', align_corners=True).squeeze().detach().numpy()

            seg_array = torch.FloatTensor(seg_array).unsqueeze(dim=0).unsqueeze(dim=0)
            seg_array = seg_array
            seg_array = F.interpolate(seg_array, (shape[0], shape[1], shape[2])).squeeze().detach().numpy()

            return ct_array, seg_array

    def randomCrop(self, input_image, input_label, crop_size=(96,96,32)):
        '''
        random crop the cubic in object region
        :param input_image:
        :param crop_size:
        :return:
        '''
        assert input_label.shape == input_image.shape,'the shape of mask and input_image should be same'
        assert isinstance(input_image,np.ndarray),'the input_image should be np.ndarray'

        # randm crop the cubic
        new_x = random.randint(0, input_image.shape[0] - crop_size[0])
        end_x = new_x + crop_size[0] - 1
        new_y = random.randint(0, input_image.shape[1] - crop_size[1])
        end_y = new_y + crop_size[1] - 1
        new_z = random.randint(0, input_image.shape[2] - crop_size[2])
        end_z = new_z + crop_size[2] - 1
        #
        image_array = input_image[new_x:end_x+1, new_y:end_y+1, new_z:end_z]
        label_array = input_label[new_x:end_x+1, new_y:end_y+1, new_z:end_z]

        return image_array, label_array

    def resample(self, input_image, target_spacing = (0.5, 0.5, 0.5)):
        '''
        resample the CT image
        :parm input_image:
        :param target_spacing: 
        '''
        assert isinstance(input_image, sitk.Image), 'the input_image should be the object of SimpleITK.SimpleITK.Image'
        origin_spacing = input_image.GetSpacing()
        origin_size = input_image.GetSize()
        scale = [target_spacing[index]/origin_spacing[index] for index in range(len(origin_size))]
        new_size = [int(origin_size[index]/scale[index]) for index in range(len(origin_size))]
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(0)
        resample.SetOutputSpacing(target_spacing)
        resample.SetOutputOrigin(input_image.GetOrigin())
        resample.SetOutputDirection(input_image.GetDirection())
        resample.SetSize(new_size)
        new_image = resample.Execute(input_image)
        return new_image
        
    def getBoundbox(self, input_array):
        '''
        get the bouding box for input_array (the non-zero range is our object)
        '''
        assert isinstance(input_array, np.ndarray)
        x,y,z = input_array.nonzero()
        return [x.min(),x.max(), y.min(), y.max(), z.min(),z.max()]

# the test code
if __name__ == '__main__':

    data_train = CTDataLoader(mode='train',use_weight=True)
    test_image, test_label = data_train.__getitem__(0)
    numbers = data_train.__len__()
    print(test_image.shape)
    print(test_label.shape)
    print('the number of cases in dataset: ', data_train.__len__())
    
