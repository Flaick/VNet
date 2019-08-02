import torch
import torch.nn as nn
from config import num_class
import torch.nn.functional as F

num_organ = num_class - 1


def get_dice(pred, organ_target):
    # the organ_target should be one-hot code
    assert len(pred.shape) == len(organ_target.shape), 'the organ_target should be one-hot code'
    dice = 0
    for organ_index in range(1, num_class):
        P = pred[:, organ_index, :, :, :]
        _P = 1 - pred[:, organ_index, :, :, :]
        G = organ_target[:, organ_index, :, :, :]
        _G = 1 - organ_target[:, organ_index, :, :, :]
        mulPG = (P * G).sum(dim=1).sum(dim=1).sum(dim=1)
        mul_PG = (_P * G).sum(dim=1).sum(dim=1).sum(dim=1)
        mulP_G = (P * _G).sum(dim=1).sum(dim=1).sum(dim=1)

        dice += (mulPG + 1) / (mulPG + 0.8 * mul_PG + 0.2 * mulP_G + 1)
    return dice

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        shape = target.shape
        organ_target = torch.zeros((target.size(0), num_organ + 1, shape[-3], shape[-2], shape[-1]))

        for organ_index in range(num_organ + 1):

            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index, :, :, :] = temp_target
            # organ_target: (B, 14, 48, 128, 128)

        organ_target = organ_target.cuda()

        # 璁＄畻绗竴闃舵鐨刲oss
        return 1-get_dice(pred, organ_target).mean()

class DiceLoss_Focal(nn.Module):
    def __init__(self, has_softmax=True):
        self.has_softmax = has_softmax
        super().__init__()

    def forward(self, pred, target):
        if self.has_softmax:
            pred = F.softmax(pred, dim=1)
        shape = target.shape
        organ_target = torch.zeros((target.size(0), num_organ + 1, shape[-3], shape[-2], shape[-1]))

        for organ_index in range(num_organ + 1):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index, :, :, :] = temp_target
            # organ_target: (B, 14, 48, 128, 128)

        organ_target = organ_target.cuda()

        # 计算第一阶段的loss
        pt_1 = get_dice(pred, organ_target).mean()
        gamma = 0.75
        return torch.pow((2-pt_1), gamma)


