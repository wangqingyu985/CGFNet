import torch
from torch import nn


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2, dilation=dilation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # channel = 1
        avg_result = torch.mean(x, dim=1, keepdim=True)  # channel = 1
        result = torch.cat([max_result, avg_result], dim=1)  # channel = 2
        output = self.conv(result)  # channel = 1
        output = self.sigmoid(output)  # channel = 1
        return output


class ChannelSpatialConv(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size, dilation=1)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        sa = self.sa(out)
        out = out * sa
        return out + residual


class MVF(nn.Module):
    def __init__(self, channel, reduction, kernels=[3, 5, 7]):
        super().__init__()
        self.k = len(kernels)
        self.convs = nn.ModuleList([])
        for kernel in kernels:
            self.convs.append(
                ChannelSpatialConv(channel=channel, reduction=reduction, kernel_size=kernel)
            )
        self.last_conv = nn.Sequential(
            convbn(in_channels=self.k * channel, out_channels=channel, kernel_size=1, stride=1, pad=0, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        residual = x  # [b, c, h, w]
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))  # k[b, c, h, w]
        feats = torch.stack(tensors=outputs, dim=1)  # [b, k, c, h, w]
        feats = feats.reshape(b, c*self.k, h, w)  # [b, k*c, h, w]
        feats = self.last_conv(feats)  # [b, c, h, w]
        return feats + residual  # [b, c, h, w]


class build_ms_volume(nn.Module):
    def __init__(self, channel, reduction, kernels=[3, 5, 7]):
        super().__init__()
        self.refAttention = MVF(channel=channel, reduction=reduction, kernels=kernels)
        self.tarAttention = MVF(channel=channel, reduction=reduction, kernels=kernels)

    def forward(self, ref_fea, tar_fea, disparity, num_groups):
        B, C, H, W = ref_fea.shape  # [B, 320, H/4, W/4]
        volume = ref_fea.new_zeros([B, num_groups, disparity, H, W])  # [B, 40, D/4, H/4, W/4]
        ref_fea = self.refAttention(ref_fea)  # [B, 320, H/4, W/4]
        tar_fea = self.tarAttention(tar_fea)  # [B, 320, H/4, W/4]
        for i in range(disparity):
            if i > 0:
                volume[:, :, i, :, i:] = groupwise_correlation(ref_fea[:, :, :, i:], tar_fea[:, :, :, :-i], num_groups)
            else:
                volume[:, :, i, :, :] = groupwise_correlation(ref_fea, tar_fea, num_groups)
        volume = volume.contiguous()  # [B, 40, D/4, H/4, W/4]
        return volume
