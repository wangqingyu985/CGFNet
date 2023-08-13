from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


def convbn_relu(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes)) 


class BasicBlock(nn.Module):  # basic resnet block
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = convbn_relu(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class BasicConv(nn.Module):  # 3d or 2d conv? conv or deconv?
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat
        if deconv and is_3d: 
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels=in_channels, out_channels=out_channels, deconv=deconv, is_3d=is_3d,
                               bn=False, relu=True, kernel_size=kernel, stride=2, padding=1)
        if self.concat:
            self.conv2 = BasicConv(in_channels=out_channels*2, out_channels=out_channels, deconv=False, is_3d=is_3d,
                                   bn=bn, relu=relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(in_channels=out_channels, out_channels=out_channels, deconv=False, is_3d=is_3d,
                                   bn=bn, relu=relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert(x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), dim=1)
        else: 
            x += rem
        x = self.conv2(x)
        return x


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn_relu(3, 32, 3, 2, 1, 1),
                                       convbn_relu(32, 32, 3, 1, 1, 1),
                                       convbn_relu(32, 32, 3, 1, 1, 1))
        self.layer1 = self._make_layer(block=BasicBlock, planes=32, blocks=1, stride=1, pad=1, dilation=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 1, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 1, 1, 1, 1)
        self.reduce = convbn_relu(in_planes=128, out_planes=32, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv1a = BasicConv(in_channels=32, out_channels=48, kernel_size=3, stride=2, padding=1)  # conv + bn + relu
        self.conv2a = BasicConv(in_channels=48, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1)

        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)

        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)  # [B, 32, H/2, W/2]
        x = self.layer1(x)  # [B, 32, H/2, W/2]
        conv0a = x  # [B, 32, H/2, W/2]
        x = self.layer2(x)  # [B, 64, H/4, W/4]
        conv1a = x  # [B, 64, H/4, W/4]
        x = self.layer3(x)  # [B, 128, H/8, W/8]
        feat0 = x  # [B, 128, H/8, W/8]
        x = self.layer4(x)  # [B, 128, H/8, W/8]
        feat1 = x  # [B, 128, H/8, W/8]
        x = self.reduce(x)  # [B, 32, H/8, W/8]
        feat2 = x  # [B, 32, H/8, W/8]
        rem0 = x  # [B, 32, H/8, W/8]
        x = self.conv1a(x)  # [B, 48, H/16, W/16]
        rem1 = x  # [B, 48, H/16, W/16]
        x = self.conv2a(x)  # [B, 64, H/32, W/32]
        rem2 = x  # [B, 64, H/32, W/32]
        x = self.conv3a(x)  # [B, 96, H/64, W/64]
        rem3 = x  # [B, 96, H/64, W/64]

        x = self.deconv3a(x, rem2)  # [B, 64, H/32, W/32]
        rem2 = x  # [B, 64, H/32, W/32]
        x = self.deconv2a(x, rem1)  # [B, 48, H/16, W/16]
        rem1 = x  # [B, 48, H/16, W/16]
        x = self.deconv1a(x, rem0)  # [B, 32, H/8, W/8]
        feat3 = x  # [B, 32, H/8, W/8]
        rem0 = x  # [B, 32, H/8, W/8]
        x = self.conv1b(x, rem1)  # [B, 48, H/16, W/16]
        rem1 = x  # [B, 48, H/16, W/16]
        x = self.conv2b(x, rem2)  # [B, 64, H/32, W/32]
        rem2 = x  # [B, 64, H/32, W/32]
        x = self.conv3b(x, rem3)  # [B, 96, H/64, W/64]
        rem3 = x  # [B, 96, H/64, W/64]
        x = self.deconv3b(x, rem2)  # [B, 64, H/32, W/32]
        x = self.deconv2b(x, rem1)  # [B, 48, H/16, W/16]
        x = self.deconv1b(x, rem0)  # [B, 32, H/8, W/8]
        feat4 = x  # [B, 32, H/8, W/8]
        gwc_feature = torch.cat((feat0, feat1, feat2, feat3, feat4), dim=1)  # [B, 128+128+32+32+32=352, H/8, W/8]
        return conv0a, gwc_feature  # [B, 32, H/2, W/2], [B, 128+128+32+32+32=352, H/8, W/8]
