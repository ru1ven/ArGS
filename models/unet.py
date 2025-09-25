#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :unet.py
#@Date        :2022/04/10 11:55:30
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr
import math

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, inplanes, outplanes, num_stages, use_final_layers=False):
        super(UNet, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.num_stages = num_stages
        self.use_final_layers = use_final_layers
        self.deconv_layers = self._make_deconv_layer(self.num_stages)
        if self.use_final_layers:
            final_layers = []
            for i in range(3):
                final_layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
                final_layers.append(nn.BatchNorm2d(self.outplanes))
                final_layers.append(nn.ReLU(inplace=True))
            self.final_layers = nn.Sequential(*final_layers)

    def _make_deconv_layer(self, num_stages):
        layers = []
        for i in range(num_stages):
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=self.outplanes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
            layers.append(nn.BatchNorm2d(self.outplanes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = self.outplanes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        if self.use_final_layers:
            x = self.final_layers(x)
        return x


ReLU = nn.ReLU
Pool = nn.MaxPool2d
BN = nn.BatchNorm2d

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            # self.relu = ReLU(out_dim)
            self.relu = ReLU(inplace=False)
        if bn:
            self.bn = BN(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()

        self.bn1 = BN(inp_dim)
        self.relu1 = ReLU(inplace=False)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = BN(int(out_dim / 2))
        self.relu2 = ReLU(inplace=False)
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = BN(int(out_dim / 2))
        self.relu3 = ReLU(inplace=False)
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        if self.need_skip:
            x = self.skip_layer(x)
        out = out + x
        return out

class AWRUnet(nn.Module):
    def __init__(self,  joint_num, deconv_dim=128, out_dim_list=[3*21, 21, 21]):
        super(AWRUnet, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]

        self.skip_layer4 = Residual(256, 256)
        self.up4 = nn.Sequential(Residual(512, 512),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual((512+256), 256)

        self.skip_layer3 = Residual(128, 128)
        self.up3 = nn.Sequential(Residual(256, 256),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((256+128), 128)

        self.skip_layer2 = Residual(64, 64)
        self.up2 = nn.Sequential(Residual(128, 128),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((128+64), deconv_dim)


        self.finals = nn.ModuleList()
        for out_dim in out_dim_list:
            self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()


    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, c1, c2, c3, c4):
        device = c1.device

        c4_up = self.up4(c4)
        c3_skip = self.skip_layer4(c3)
        c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(c2)
        c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(c1)
        img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1))

        img_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(img_feature)
            img_result = torch.cat((img_result, temp), dim=1)

        return img_result, img_feature