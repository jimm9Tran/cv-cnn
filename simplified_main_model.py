#coding=utf-8
from __future__ import absolute_import
from __future__ import division
import torch
from torch import nn
from torch.nn import functional as F
from senet import senet154


class ConvBlock(nn.Module):
    """Basic convolutional block.
    
    convolution + batch normalization + relu.
    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SpatialAttn(nn.Module):
    """
    空间上的attention 模块.
    Spatial Attention (Sec. 3.1.I.1)"""

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)  # 由hwc 变为 hw1
        # 3-by-3 conv
        h = x.size(2)
        x = self.conv1(x)
        # bilinear resizing
        x = F.upsample(x, (h, h), mode='bilinear', align_corners=True)
        # scaling conv
        x = self.conv2(x)
        ## 返回的是h*w*1 的 soft map
        return x


class ChannelAttn(nn.Module):
    """
    通道上的attention 模块
    Channel Attention (Sec. 3.1.I.2)
    """

    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])  #输出是1*1*c
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        # 返回的是 1*1*c 的map
        return x


class SoftAttn(nn.Module):
    """
    Soft Attention (Sec. 3.1.I)
    空间和通道上的attention 融合
    就是空间和通道上的attention做一个矩阵乘法
    Aim: Spatial Attention + Channel Attention
    
    Output: attention maps with shape identical to input.
    """

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        # 返回的是 hwc
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y


class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""

    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, 4 * 2)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        # 初始化 参数
        # if x_t = 0  the performance is very low
        self.fc.bias.data.copy_(
            torch.tensor([0.3, -0.3, 0.3, 0.3, -0.3, 0.3, -0.3, -0.3],
                         dtype=torch.float))

    def forward(self, x):
        '''
        输出的是STN 需要的theta
        '''
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        # predict transformation parameters
        theta = torch.tanh(self.fc(x))
        theta = theta.view(-1, 4, 2)
        #  返回的是 2T  T为区域数量。 因为尺度会固定。 所以只要学位移的值
        return theta


class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""

    def __init__(self, in_channels):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)
        self.hard_attn = HardAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        theta = self.hard_attn(x)
        return y_soft_attn, theta


class SENet_BASE(nn.Module):

    def __init__(
        self,
        num_classes,
        senet154_weight,
        multi_scale=False,
    ):
        super(SENet_BASE, self).__init__()
        self.senet154_weight = senet154_weight
        self.num_classes = num_classes

        # construct SEnet154
        senet154_ = senet154(num_classes=1000, pretrained=None)
        senet154_.load_state_dict(torch.load(self.senet154_weight))
        self.main_network = nn.Sequential(senet154_.layer0, senet154_.layer1,
                                          senet154_.layer2, senet154_.layer3,
                                          senet154_.layer4,
                                          nn.AdaptiveAvgPool2d((1, 1)))

        self.global_out = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(2048, num_classes))

    def forward(self, x):
        x = self.main_network(x)
        x = x.view(x.size(0), -1)
        return self.global_out(x)


class MODEL(nn.Module):
    '''
    the mian model of TII2020
    '''

    def __init__(self,
                 num_classes,
                 senet154_weight,
                 nchannels=[256, 512, 1024, 2048],
                 multi_scale=False):
        super(MODEL, self).__init__()
        self.senet154_weight = senet154_weight
        self.multi_scale = multi_scale
        self.num_classes = num_classes

        # construct SEnet154
        senet154_ = senet154(num_classes=1000, pretrained=None)
        senet154_.load_state_dict(torch.load(self.senet154_weight))
        self.global_layers = nn.Sequential(
            senet154_.layer0,
            senet154_.layer1,
            senet154_.layer2,
            senet154_.layer3,
            senet154_.layer4,
        )
        self.ha_layers = nn.Sequential(
            HarmAttn(nchannels[1]),
            HarmAttn(nchannels[2]),
            HarmAttn(nchannels[3]),
        )
        if self.multi_scale:
            self.global_out = nn.Sequential(
                nn.Linear(nchannels[1] + nchannels[2] + nchannels[3],
                          num_classes))
        else:
            self.global_out = nn.Sequential(nn.Dropout(0.2),
                                            nn.Linear(2048, num_classes))

    def forward(self, x):
        """
        add the mutli-scal method 10/17.  
        first how is the fusion global and local recognition performance  
        second add the multi-scal method  
        where added? oral output or attention's output?  
        """

        def msa_block(inp, main_conv, harm_attn):
            inp = main_conv(inp)
            inp_attn, _ = harm_attn(inp)
            return inp_attn * inp

        x = self.global_layers[0](x)
        x1 = self.global_layers[1](x)
        # 512*28*28
        x2 = msa_block(x1, self.global_layers[2], self.ha_layers[0])
        # 1024*14*14
        x3 = msa_block(x2, self.global_layers[3], self.ha_layers[1])
        # 2048*7*7
        x4 = msa_block(x3, self.global_layers[4], self.ha_layers[2])

        #全局pooling 2048
        x4_avg = F.avg_pool2d(x4, x4.size()[2:]).view(x4.size(0), -1)

        if self.multi_scale:
            #512 向量
            x2_avg = F.adaptive_avg_pool2d(x2, (1, 1)).view(x2.size(0), -1)
            #1024 向量
            x3_avg = F.adaptive_avg_pool2d(x3, (1, 1)).view(x3.size(0), -1)

            multi_scale_feature = torch.cat([x2_avg, x3_avg, x4_avg], 1)
            global_out = self.global_out(multi_scale_feature)

        else:
            global_out = self.global_out(
                self.dropout(x4_avg))  # 2048 -> num_classes

        return global_out