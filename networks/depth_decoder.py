# # Copyright Niantic 2019. Patent Pending. All rights reserved.
# #
# # This software is licensed under the terms of the Monodepth2 licence
# # which allows for non-commercial use only, the full terms of which are made
# # available in the LICENSE file.
#

from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *
# from .hr_layers import *
from timm.models.layers import trunc_normal_
from scale_casa import scale_casa_HAM

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda(0).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ECCAttention(nn.Module):
    def __init__(self, in_channels):
        super(ECCAttention, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 8
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        self.SoftMax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.ema = eca_layer(224)

    def forward(self, x):
        b, _, h, w = x.size()
        # [b, c', h, w]
        query = self.ConvQuery(x)
        # [b, w, c', h] -> [b*w, c', h] -> [b*w, h, c']
        query_H = query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        # [b, h, c', w] -> [b*h, c', w] -> [b*h, w, c']
        query_W = query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)

        # [b, c', h, w]
        key = self.ConvKey(x)
        # [b, w, c', h] -> [b*w, c', h]
        key_H = key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        # [b, h, c', w] -> [b*h, c', w]
        key_W = key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # [b, c, h, w]
        value = self.ConvValue(x)
        # [b, w, c, h] -> [b*w, c, h]
        value_H = value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        # [b, h, c, w] -> [b*h, c, w]
        value_W = value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # [b*w, h, c']* [b*w, c', h] -> [b*w, h, h] -> [b, h, w, h]
        energy_H = (torch.bmm(query_H, key_H) + self.INF(b, h, w)).view(b, w, h, h).permute(0, 2, 1, 3)
        # [b*h, w, c']*[b*h, c', w] -> [b*h, w, w] -> [b, h, w, w]
        energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)
        # [b, h, w, h+w]  concate channels in axis=3
        concate = self.SoftMax(torch.cat([energy_H, energy_W], 3))

        # [b, h, w, h] -> [b, w, h, h] -> [b*w, h, h]
        attention_H = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
        # [b*h, w, w]
        attention_W = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w)

        # [b*w, h, c]*[b*w, h, h] -> [b, w, c, h] error [b,c,h,w]
        out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        # [b,c,h,w]
        out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)
        chanel = self.ema(x)
        return self.gamma * (out_H + out_W + chanel) + x



class PWSA(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(PWSA, self).__init__()

        self.Conv1x1 = Conv1x1(input_channel, input_channel)
        self.Res_block = ConvBlock(input_channel, input_channel)
        self.softmax = nn.Softmax(dim=1)
        # self.upsample = upsample()
        self.Conv1x1_out = Conv1x1(input_channel, output_channel)

    def forward(self, FD, FE):
        Sadd = (FD + FE) / 2
        Satt = self.Res_block(FE)
        Satt = self.softmax(self.Conv1x1(Satt))
        Sscaled = Sadd * Satt
        S = self.Res_block(Sscaled)
        FD_out = upsample(self.Conv1x1_out(S))

        return FD_out

class fSEModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(fSEModule, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        # print(
        #     f"DEBUG: high={high_feature_channel}, low={low_feature_channels}, sum={in_channel}")
        if output_channel is not None:
            out_channel = output_channel
        reduction = 16
        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features, low_features):
        # features = [upsample(high_features)]
        # features += low_features
        # features = torch.cat(features, 1)
        high_upsampled = upsample(high_features)
        features = torch.cat([high_upsampled, low_features], dim=1)

        b, c, _, _ = features.size()
        y = self.avg_pool(features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        y = self.sigmoid(y)
        features = features * y.expand_as(features)

        return self.relu(self.conv_se(features))
class NWCHead(nn.Module):
    def __init__(self, in_channels, k=3):
        """
        NWC 预测头

        参数:
            in_channels (int): 输入特征图的通道数
            k (int): 卷积核大小，默认 3
        """
        super(NWCHead, self).__init__()
        self.k = k
        self.k_sq = k * k

        # 深度向量预测卷积层
        self.conv_depth = nn.Conv2d(in_channels, self.k_sq, kernel_size=k, padding=k // 2)

        # 置信度向量预测卷积层
        self.conv_confidence = nn.Conv2d(in_channels, self.k_sq, kernel_size=k, padding=k // 2)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv_depth.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv_depth.bias, 0)
        nn.init.kaiming_normal_(self.conv_confidence.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv_confidence.bias, 0)

    def forward(self, F_up):
        """
        前向传播

        参数:
            F_up (torch.Tensor): 上采样后的特征图，形状为 [batch_size, in_channels, height, width]

        返回:
            D (torch.Tensor): 最终深度估计，形状为 [batch_size, 1, height, width]
        """
        batch_size, _, height, width = F_up.size()

        # 预测深度向量 V
        V = self.conv_depth(F_up)  # [batch_size, k_sq, height, width]
        V = torch.sigmoid(V)  # 使用 Sigmoid 激活函数

        # 预测置信度向量 P
        P = self.conv_confidence(F_up)  # [batch_size, k_sq, height, width]
        P = F.softmax(P, dim=1)  # 使用 Softmax 激活函数，使得 P 在 k_sq 维度上和为 1

        # 计算最终深度 D
        # Reshape V 和 P 为 [batch_size, k_sq, height * width]
        V = V.view(batch_size, self.k_sq, -1)  # [batch_size, k_sq, height * width]
        P = P.view(batch_size, self.k_sq, -1)  # [batch_size, k_sq, height * width]

        # 逐元素相乘并在 k_sq 维度上求和
        D = torch.sum(V * P, dim=1, keepdim=True)  # [batch_size, 1, height * width]
        D = D.view(batch_size, 1, height, width)  # [batch_size, 1, height, width]

        return D


# class FusionDecoder(nn.Module):
#     def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
#         super(FusionDecoder, self).__init__()
#
#         self.num_output_channels = num_output_channels
#         self.scales = scales
#
#         self.num_ch_enc = num_ch_enc  # features in encoder, [64, 18, 36, 72, 144][16,64,128,160,320]
#         # self.num_ch_enc[0] = 96
#
#         # decoder
#         self.convs = OrderedDict()
#         # self.ecca1 = ECCAttention(self.num_ch_enc[-1])
#
#
#         self.convs[("parallel_conv"), 2, 0] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
#         self.convs[("parallel_conv"), 2, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
#         self.convs[("parallel_conv"), 2, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
#
#
#
#         self.convs[("conv1x1", 2, 2_1)] = ConvBlock1x1(self.num_ch_enc[2] + self.num_ch_enc[1], self.num_ch_enc[1])
#         self.convs[("conv1x1", 2, 2_0)] = ConvBlock1x1(self.num_ch_enc[0] + self.num_ch_enc[1], self.num_ch_enc[1])
#         self.convs[("conv1x1", 2, 3_0)] = ConvBlock1x1(self.num_ch_enc[0] + self.num_ch_enc[1], self.num_ch_enc[1])
#         self.convs[("attention", 2)] = fSEModule(self.num_ch_enc[2], self.num_ch_enc[3])
#         self.convs[("attention", 1)] = fSEModule(self.num_ch_enc[1], self.num_ch_enc[2])
#         self.convs[("attention", 0)] = fSEModule(self.num_ch_enc[0], self.num_ch_enc[1])
#
#         self.convs[("parallel_conv"), 3, 0] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
#         self.convs[("parallel_conv"), 3, 1] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
#
#         self.convs[("attention", 1)] = fSEModule(self.num_ch_enc[1], self.num_ch_enc[2])
#
#         self.convs[("up_conv"), 0] = ConvBlock(96, 48)
#
#         self.convs[("dispconv", 0)] = Conv3x3(48, self.num_output_channels)
#
#         self.decoder = nn.ModuleList(list(self.convs.values()))
#         self.sigmoid = nn.Sigmoid()
#         self.CASA = scale_casa_HAM([48, 48, 80, 128])
#
#
#     def FusionConv(self, conv, high_feature, low_feature):
#
#         high_features = [upsample(high_feature)]
#         high_features.append(low_feature)
#         high_features = torch.cat(high_features, 1)
#
#         return conv(high_features)
#
#     def FusionConv_nosample(self, conv, high_feature, low_feature):
#         high_features = [high_feature]
#         high_features.append(low_feature)
#         high_features = torch.cat(high_features, 1)
#
#         return conv(high_features)
#
#     def forward(self, input_features):
#         self.outputs = {}
#         # input_features = self.CASA(input_features)
#
#         e = updown_sample(input_features[0], 2)#48
#
#         # e2 = input_features[3]
#         e2 = self.ecca1(input_features[3])#128
#         e1 = input_features[2]#80
#         e0 = input_features[1]#48
#
#         d2_0 = self.convs[("parallel_conv"), 2, 0](e)#48
#         d2_1 = self.convs[("parallel_conv"), 2, 1](e0)#48
#         d2_2 = self.convs[("parallel_conv"), 2, 2](e1)#80
#
#         d23_2 = self.convs[("attention", 2)](e2, d2_2)#80
#         d22_1 = self.FusionConv(self.convs[("conv1x1", 2, 2_1)], d2_2, d2_1)#48
#         d22_0 = self.FusionConv(self.convs[("conv1x1", 2, 2_0)], d2_1, d2_0)#48
#
#         d3_1 = self.convs[("parallel_conv"), 3, 1](d22_1)#48
#         d3_0 = self.convs[("parallel_conv"), 3, 0](d22_0)#48
#
#         d32_1 = self.convs[("attention", 1)](d23_2, d3_1)#80
#         d12_0 = self.FusionConv(self.convs[("conv1x1", 2, 3_0)], d3_1, d3_0)#48
#         d3_0 = self.convs[("parallel_conv"), 4, 0](d12_0)
#         d32_1 = self.convs[("attention", 0)](d32_1, d3_0)
#
#         d = self.convs[("parallel_conv"), 3, 0](d32_1)
#
#         d = [updown_sample(d, 2)]
#         d += [e]
#         d = torch.cat(d, 1)
#         d = self.convs[("up_conv"), 0](d)
#
#
#         self.outputs[("disp", 0)] = self.sigmoid(updown_sample(self.convs[("dispconv", 0)](d), 2))
#
#         return self.outputs  # single-scale depth
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Fully Connected Layers
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: Residual connection (re-weighting)
        return x * y.expand_as(x)
class FusionDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(FusionDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales

        self.num_ch_enc = num_ch_enc  # features in encoder, [64, 18, 36, 72, 144][16,64,128,160,320]
        # self.num_ch_enc[0] = 96

        # decoder
        self.convs = OrderedDict()
        self.ecca1 = ECCAttention(self.num_ch_enc[-1])


        self.convs[("parallel_conv"), 2, 0] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 2, 00] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 2, 1] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])



        self.convs[("conv1x1", 2, 2_1)] = ConvBlock1x1(self.num_ch_enc[2] + self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("attention", 2)] = fSEModule(self.num_ch_enc[2], self.num_ch_enc[3])
        self.convs[("attention", 111)] = fSEModule(self.num_ch_enc[1], self.num_ch_enc[3])

        self.convs[("parallel_conv"), 3, 0] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 3, 10] = ConvBlock(96, 96)

        self.convs[("attention", 1)] = fSEModule(self.num_ch_enc[1], self.num_ch_enc[2])

        self.convs[("attention", 000)] = fSEModule(self.num_ch_enc[1], self.num_ch_enc[1])

        self.convs[("up_conv"), 0] = ConvBlock(144, 48)

        self.convs[("dispconv", 0)] = Conv3x3(48, self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()


    def FusionConv(self, conv, high_feature, low_feature):

        high_features = [upsample(high_feature)]
        high_features.append(low_feature)
        high_features = torch.cat(high_features, 1)

        return conv(high_features)

    def FusionConv_nosample(self, conv, high_feature, low_feature):
        high_features = [high_feature]
        high_features.append(low_feature)
        high_features = torch.cat(high_features, 1)

        return conv(high_features)

    def forward(self, input_features):
        self.outputs = {}

        e = updown_sample(input_features[0], 2)#48

        e2 = self.ecca1(input_features[3]) #128
        e1 = input_features[2] #80
        e0 = input_features[1]  #48
        e00 = updown_sample(e0, 0.5)   #48
        d2_1 = self.convs[("parallel_conv"), 2, 0](e0)#48
        d2_2 = self.convs[("parallel_conv"), 2, 1](e1)#80
        d0_0 = self.convs[("parallel_conv"), 2, 00](e00)#48



        d23_2 = self.convs[("attention", 2)](e2, d2_2)
        d23_00 = self.convs[("attention", 111)](e2, d0_0)
        d22_1 = self.FusionConv(self.convs[("conv1x1", 2, 2_1)], d2_2, d2_1)

        d3_0 = self.convs[("parallel_conv"), 3, 0](d22_1)
        d32_1 = self.convs[("attention", 1)](d23_2, d3_0)#48
        d32_00 = self.convs[("attention", 000)](d23_00, d3_0)#48

        fused = torch.cat([d32_1, d32_00], dim=1)
        d = self.convs[("parallel_conv"), 3, 10](fused)

        d = [updown_sample(d, 2)]
        d += [e]
        d = torch.cat(d, 1)
        d = self.convs[("up_conv"), 0](d)


        self.outputs[("disp", 0)] = self.sigmoid(updown_sample(self.convs[("dispconv", 0)](d), 2))

        return self.outputs  # single-scale depth
