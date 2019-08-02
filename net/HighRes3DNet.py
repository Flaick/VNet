import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# https://link.springer.com/content/pdf/10.1007%2F978-3-319-59050-9_28.pdf
#
class HighRes3DNet(nn.Module):
    """
    共9332094个可训练的参数, 九百三十万左右
    """
    def __init__(self, inchannel,num_organ = 2, softmax = False):
        super().__init__()

        self.inchannel = inchannel
        # self.training = training
        # self.stage = stage
        self.num_organ = num_organ
        self.softmax = softmax

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(inchannel, 16, 3, 1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )
        ###
        self.encoder_stage2 = nn.Sequential(
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, 1, padding=1),

            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, 1, padding=1),
        )
        self.encoder_stage3 = nn.Sequential(
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, 1, padding=1),

            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, 1, padding=1),
        )
        self.encoder_stage4 = nn.Sequential(
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, 1, padding=1),

            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, 1, padding=1),
        )
        ######
        self.upchannel1 = nn.Conv3d(16,32,1)
        self.encoder_stage5 = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, padding=2,dilation=2),

            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, padding=2,dilation=2),
        )
        self.encoder_stage6 = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, padding=2,dilation=2),

            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, padding=2,dilation=2),
        )
        self.encoder_stage7 = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, padding=2,dilation=2),

            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, padding=2,dilation=2),
        )
        ####
        self.upchannel2 = nn.Conv3d(32,64,1)
        self.encoder_stage8 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, 1, padding=4,dilation=4),

            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, 1, padding=4,dilation=4),
        )
        self.encoder_stage9 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, 1, padding=4,dilation=4),

            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, 1, padding=4,dilation=4),
        )
        self.encoder_stage10 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, 1, padding=4,dilation=4),

            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, 1, padding=4,dilation=4),
        )
        if self.softmax:
            self.map = nn.Sequential(
                nn.Conv3d(64, self.num_organ + 1, 1),
                nn.Softmax(dim=1)
            )
        else:
            self.map = nn.Conv3d(64, self.num_organ+1, 1)

    def forward(self, inputs):
        stage1 = self.encoder_stage1(inputs)

        stage2 = self.encoder_stage2(stage1) + stage1

        stage3 = self.encoder_stage3(stage2) + stage2

        stage4 = self.encoder_stage4(stage3) + stage3

        stage5_upchannel = self.upchannel1(stage4)
        stage5 = self.encoder_stage5(stage5_upchannel) + stage5_upchannel

        stage6 = self.encoder_stage6(stage5) + stage5

        stage7 = self.encoder_stage7(stage6) + stage6

        stage8_upchannel = self.upchannel2(stage7)
        stage8 = self.encoder_stage8(stage8_upchannel) + stage8_upchannel

        stage9 = self.encoder_stage9(stage8) + stage8

        stage10 = self.encoder_stage10(stage9) + stage9

        outputs = self.map(stage10)

        return outputs

# 网络参数初始化函数
def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)

if __name__ == '__main__':
    net = HighRes3DNet(training=True, inchannel=1, stage='stage1')
    net.apply(init)

    # 输出数据维度检查
    net = net.cuda()
    data = torch.randn((1, 1, 16, 96, 96)).cuda()

    with torch.no_grad():
        res = net(data)

    print(res.shape)




