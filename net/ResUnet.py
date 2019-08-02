"""
ResUNet整体上网络结构基于VNet，做出的修改如下：
将原本555的卷积核换成了333
在除了第一个和最后一个block中添加了dropout
去掉了编码器部分的最后一个16倍降采样的stage
为了弥补这么做带来的感受野的损失，在编码器的最后两个stage加入了混合空洞卷积
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import config
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# 定义单个3D FCN
class ResUNet(nn.Module):
    """
    共9332094个可训练的参数, 九百三十万左右
    """
    def __init__(self, training, inchannel, stage, dropout_rate = 0.3, withLogits = False):
        """
        :param training: 标志网络是属于训练阶段还是测试阶段
        :param inchannel 网络最开始的输入通道数量
        :param stage 标志网络属于第一阶段，还是第二阶段
        :param num_organ: the numbers without the background class
        :param withLogits: whether the output should be processed by the activation (softmax)
        """
        super().__init__()

        self.training = training
        self.stage = stage
        self.num_organ = config.num_class - 1
        self.dropout_rate = dropout_rate
        self.withLogits = withLogits
        
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(inchannel, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        if self.withLogits: # 
            self.map = nn.Sequential(
            nn.Conv3d(32, self.num_organ + 1, 1),
            nn.Softmax(dim=1))
        else:
            self.map = nn.Conv3d(32, self.num_organ + 1, 1)

    def forward(self, inputs):

        if self.stage is 'stage1':
            long_range1 = self.encoder_stage1(inputs) + inputs
        else:
            long_range1 = self.encoder_stage1(inputs)

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, self.dropout_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, self.dropout_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, self.dropout_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, self.dropout_rate, self.training)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, self.dropout_rate, self.training)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, self.dropout_rate, self.training)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        outputs = self.map(outputs)

        # 返回概率图
        return outputs

# 定义最终的级连3D FCN
class StageNet(nn.Module):
    def __init__(self, training, detach = False):
        super().__init__()

        self.training = training
        self.detach = detach # whether detach the gradient of stage1
        self.stage1 = ResUNet(training=training, inchannel=1, stage='stage1')#,num_organ=config.num_class-1)
        self.stage2 = ResUNet(training=training, inchannel=config.num_class + 1, stage='stage2')#,num_organ=config.num_class-1)

    def forward(self, inputs, cube_glob, start_slice, end_slice):
        """
        首先将输入数据在轴向上缩小一倍，然后送入第一阶段网络中
        得到一个粗糙尺度下的分割结果
        然后将原始尺度大小的数据与第一步中得到的分割结果进行拼接，共同送入第二阶段网络中
        得到最终的分割结果
        共18656348个可训练的参数，一千八百万左右
        """
        cube_glob = cube_glob.float()
        shape = cube_glob.shape # [1,1,80,384, 240]
        # 首先将输入缩小一倍
        # inputs_stage1 = F.upsample(inputs, (48, 128, 128), mode='trilinear')
        inputs_stage1 = nn.functional.interpolate(cube_glob, (shape[2], shape[3]//2, shape[4]//2), mode='trilinear', align_corners=True)# [1,1,80,192, 120]

        # 得到第一阶段的结果
        #inputs_stage1 = cube_glob
        if self.detach:
            output_stage1 = self.stage1(inputs_stage1.detach())
        else:
            output_stage1 = self.stage1(inputs_stage1) # [1, 3, 32, 192, 120]
        # output_stage1 = F.upsample(output_stage1, (48, 256, 256), mode='trilinear')
        output_stage1 = F.interpolate(output_stage1, (shape[2], shape[3], shape[4]), mode='trilinear', align_corners=True) # shape[2]

        temp = F.softmax(output_stage1, dim=1)
        # 将第一阶段的结果与原始输入数据进行拼接作为第二阶段的输入
        inputs_stage2 = torch.cat((temp[:,:,start_slice:end_slice + 1,:,:], inputs), dim=1) #[1, 4, 32, 384, 240]

        # 得到第二阶段的结果
        output_stage2 = self.stage2(inputs_stage2) # [1, 3, 32, 384, 240]

        if self.training is True:
            return output_stage1, output_stage2
        else:
            return output_stage1, output_stage2


# 网络参数初始化函数
def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)
#
# the test code
if __name__ == '__main__':
    net = DeepResUNet(training=True, inchannel=1, stage='stage1')
    net.apply(init)

    # 输出数据维度检查
    net = net.cuda()
    data = torch.randn((1, 1, 32, 384, 240)).cuda()

    with torch.no_grad():
        res = net(data)

    for item in res:
        print(item.size())

    # 计算网络参数
    num_parameter = .0
    for item in net.modules():

        if isinstance(item, nn.Conv3d) or isinstance(item, nn.ConvTranspose3d):
            num_parameter += (item.weight.size(0) * item.weight.size(1) *
                              item.weight.size(2) * item.weight.size(3) * item.weight.size(4))

            if item.bias is not None:
                num_parameter += item.bias.size(0)

        elif isinstance(item, nn.PReLU):
            num_parameter += item.num_parameters
    print(num_parameter)



