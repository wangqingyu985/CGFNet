from __future__ import print_function
from models.feature_extractor_fast import feature_extraction
from models.MVF import build_ms_volume
from models.submodules3d import CoeffsPredictor
from models.submodules2d import HourglassRefinement
from models.submodules import SubModule, convbn_2d_lrelu, convbn_3d_lrelu, convbn_2d_Tanh, convbn_3d, convbn
import torch
import torch.nn as nn
import torch.nn.functional as F


class Slice(SubModule):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, wg, hg, guidemap): 
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W]-> [B, H, W, C]
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=False)
        return coeff.squeeze(2)  # [B, 1, H, W]


class GuideNN(SubModule):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = convbn_2d_lrelu(in_planes=32, out_planes=16, kernel_size=1, stride=1, pad=0)
        self.conv2 = convbn_2d_Tanh(in_planes=16, out_planes=1, kernel_size=1, stride=1, pad=0)

    def forward(self, x):  # [B, 32, H/2, W/2]
        return self.conv2(self.conv1(x))  # [B, 1, H/2, W/2]


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])  # [B, G, D, H, W]
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def correlation(fea1, fea2):
    B, C, H, W = fea1.shape
    cost = (fea1 * fea2).mean(dim=1)
    assert cost.shape == (B, H, W)
    return cost


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
                                   nn.BatchNorm3d(in_channels * 2),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                                   nn.BatchNorm3d(in_channels * 2),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv3d(in_channels=in_channels * 2, out_channels=in_channels * 4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
                                   nn.BatchNorm3d(in_channels * 4),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv3d(in_channels=in_channels * 4, out_channels=in_channels * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                                   nn.BatchNorm3d(in_channels * 4),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels * 4, out_channels=in_channels * 2, kernel_size=(3, 3, 3), padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(in_channels * 2))
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=(3, 3, 3), padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(in_channels))
        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


class CGFNet(SubModule):
    def __init__(self, maxdisp, cnn_weights_only, freeze_cnn_weights):
        super(CGFNet, self).__init__()
        self.maxdisp = maxdisp
        self.cnn_weights_only = cnn_weights_only
        self.freeze_cnn_weights = freeze_cnn_weights
        self.concat_channels = 32

        self.softmax = nn.Softmax(dim=1)

        self.feature_extraction = feature_extraction()
        self.build = build_ms_volume(channel=352, reduction=4, kernels=[3, 5, 7])
        self.coeffs_disparity_predictor = CoeffsPredictor()
        self.refinement_net = HourglassRefinement()

        self.dres1_att = nn.Sequential(convbn_3d(44, 16, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn_3d(16, 16, 3, 1, 1))
        self.dres2_att = hourglass(16)
        self.classif_att = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.concatconv = nn.Sequential(convbn(in_planes=352, out_planes=128,
                                               kernel_size=3, stride=1, pad=1, dilation=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=128, out_channels=self.concat_channels,
                                                  kernel_size=1, padding=0, stride=1, bias=False)
                                        )
        self.dres0 = nn.Sequential(convbn_3d_lrelu(in_planes=44, out_planes=32, kernel_size=3, stride=1, pad=1),
                                   convbn_3d_lrelu(in_planes=32, out_planes=16, kernel_size=3, stride=1, pad=1))
        self.guide = GuideNN()
        self.slice = Slice()
        self.weight_init()

    def forward(self, left_input, right_input):

        if self.freeze_cnn_weights:
            with torch.no_grad():
                left_low_level_features_1, left_gwc_feature = self.feature_extraction(left_input)
                # [B, 32, H/2, W/2], [B, 128+128+32+32+32=352, H/8, W/8]
                _, right_gwc_feature = self.feature_extraction(right_input)  # [B, 128+128+32+32+32=352, H/8, W/8]

                gwc_volume = build_gwc_volume(refimg_fea=left_gwc_feature,
                                              targetimg_fea=right_gwc_feature,
                                              maxdisp=25,
                                              num_groups=44)  # [B, 44, 25, H/8, W/8]

                cost_cnn = self.dres1_att(gwc_volume)  # [B, 16, 25, H/8, W/8]
                cost_cnn = self.dres2_att(cost_cnn)  # [B, 16, 25, H/8, W/8]
                cnn_weights = self.classif_att(cost_cnn)  # [B, 1, 25, H/8, W/8]
        else:
            left_low_level_features_1, left_gwc_feature = self.feature_extraction(left_input)
            # [B, 32, H/2, W/2], [B, 128+128+32+32+32=352, H/8, W/8]
            _, right_gwc_feature = self.feature_extraction(right_input)  # [B, 128+128+32+32+32=352, H/8, W/8]

            gwc_volume = build_gwc_volume(refimg_fea=left_gwc_feature,
                                          targetimg_fea=right_gwc_feature,
                                          maxdisp=25,
                                          num_groups=44)  # [B, 44, 25, H/8, W/8]

            cost_cnn = self.dres1_att(gwc_volume)  # [B, 16, 25, H/8, W/8]
            cost_cnn = self.dres2_att(cost_cnn)  # [B, 16, 25, H/8, W/8]
            cnn_weights = self.classif_att(cost_cnn)  # [B, 1, 25, H/8, W/8]

        if not self.cnn_weights_only:
            guide = self.guide(left_low_level_features_1)  # [B, 1, H/2, W/2]
            cost_volume = self.build(ref_fea=left_gwc_feature, tar_fea=right_gwc_feature, disparity=25, num_groups=44)
            # cost_volume = build_gwc_volume(refimg_fea=left_gwc_feature, targetimg_fea=right_gwc_feature, maxdisp=25, num_groups=44)  # [B, 44, 25, H/8, W/8]
            ac_volume = F.softmax(input=cnn_weights, dim=2) * cost_volume  # [B, 44, 25, H/8, W/8]
            cost_volume = self.dres0(ac_volume)  # [B, 16, 25, H/8, W/8]
            coeffs = self.coeffs_disparity_predictor(cost_volume)  # [B, 25, 32, H/8, W/8]
            list_coeffs = torch.split(tensor=coeffs, split_size_or_sections=1, dim=1)  # 25 * [B, 1, 32, H/8, W/8]
            index = torch.arange(0, 97)  # [97]: 0,1,2,3,4...96
            index_float = index/4.0  # [97]: 0,0.25,0.5,0.75,1...24
            index_a = torch.floor(input=index_float)  # [97]: 0,0,0,0,1,1,1,1...24
            index_b = index_a + 1  # [97]: 1,1,1,1,2,2,2,2...24
            index_a = torch.clamp(input=index_a, min=0, max=24)  # [97]: 0,0,0,0,1,1,1,1...24
            index_b = torch.clamp(input=index_b, min=0, max=24)  # [97]: 1,1,1,1,2,2,2,2...24
            wa = index_b - index_float  # [97]: 1,0.75,0.5,0.25,1,0.75,0.5,0.25...
            wb = index_float - index_a  # [97]: 0,0.25,0.5,0.75,0,0.25,0.5,0.75...
            list_float = []
            device = list_coeffs[0].get_device()
            wa = wa.view(1, -1, 1, 1)  # [1, 97, 1, 1]
            wb = wb.view(1, -1, 1, 1)  # [1, 97, 1, 1]
            wa = wa.to(device)  # [1, 97, 1, 1]
            wb = wb.to(device)  # [1, 97, 1, 1]
            wa = wa.float()  # [1, 97, 1, 1]
            wb = wb.float()  # [1, 97, 1, 1]
            N, _, H, W = guide.shape  # [B, _, H/2, W/2]
            hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [H/2, W/2], [H/2, W/2]
            if device >= 0:
                hg = hg.to(device)  # [H/2, W/2]
                wg = wg.to(device)  # [H/2, W/2]
            # [B,H,W,1]
            hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1  # [B, H/2, W/2, 1], norm to [-1,1]
            wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1  # [B, H/2, W/2, 1], norm to [-1,1]
            slice_dict = []
            for i in range(25):
                slice_dict.append(self.slice(list_coeffs[i], wg, hg, guide))  # 25 * [B, 1, H/2, W/2]
            slice_dict_a = []  # 97 * [B, 1, H/2, W/2]
            slice_dict_b = []  # 97 * [B, 1, H/2, W/2]
            for i in range(97):
                inx_a = i//4
                inx_b = inx_a + 1
                inx_b = min(inx_b, 24)
                slice_dict_a.append(slice_dict[inx_a])
                slice_dict_b.append(slice_dict[inx_b])
# ------------------------------------------------------------------- End of building network structure.
        if self.training:
            if not self.freeze_cnn_weights:
                cost_cnn = F.upsample(cnn_weights, [self.maxdisp, left_input.size()[2], left_input.size()[3]], mode='trilinear')  # [B, 1, 192, H, W]
                cost_cnn = torch.squeeze(cost_cnn, 1)  # [B, 192, H, W]
                pred_cnn = F.softmax(cost_cnn, dim=1)  # [B, 192, H, W]
                out1 = disparity_regression(x=pred_cnn, maxdisp=self.maxdisp)  # [B, 1, H, W]
                out1 = torch.squeeze(out1, 1)  # [B, H, W]

            if not self.cnn_weights_only:
                final_cost_volume = wa * torch.cat(slice_dict_a, dim=1) + wb * torch.cat(slice_dict_b,
                                                                                         dim=1)  # [B, 97, H/2, W/2]
                slice = self.softmax(final_cost_volume)  # [B, 97, H/2, W/2]
                disparity_samples = torch.arange(0, 97, dtype=slice.dtype, device=slice.device).view(1, 97, 1,
                                                                                                     1)  # [1, 97, 1, 1]
                disparity_samples = disparity_samples.repeat(slice.size()[0], 1, slice.size()[2],
                                                             slice.size()[3])  # [B, 97, H/2, W/2]
                half_disp = torch.sum(disparity_samples * slice, dim=1).unsqueeze(1)  # [B, 1, H/2, W/2]
                left_half = F.interpolate(
                    left_input,
                    scale_factor=1 / pow(2, 1),
                    mode='bilinear',
                    align_corners=False)
                right_half = F.interpolate(
                    right_input,
                    scale_factor=1 / pow(2, 1),
                    mode='bilinear',
                    align_corners=False)
                refinement_disp = self.refinement_net(half_disp, left_half, right_half)
                out2 = F.interpolate(refinement_disp * 2.0, scale_factor=(2.0, 2.0),
                                     mode='bilinear', align_corners=False).squeeze(1)  # [B, H, W]

                if self.freeze_cnn_weights:
                    return [out2]
                return [out1, out2]
            return [out1]  # [B, H, W]

        else:
            if self.cnn_weights_only:
                cost_cnn = F.upsample(cnn_weights, [self.maxdisp, left_input.size()[2], left_input.size()[3]], mode='trilinear')  # [B, 1, 192, H, W]
                cost_cnn = torch.squeeze(cost_cnn, 1)  # [B, 192, H, W]
                pred_cnn = F.softmax(cost_cnn, dim=1)  # [B, 192, H, W]
                out1 = disparity_regression(x=pred_cnn, maxdisp=self.maxdisp)  # [B, 1, H, W]
                out1 = torch.squeeze(out1, 1)  # [B, H, W]
                return [out1]

            final_cost_volume = wa * torch.cat(slice_dict_a, dim=1) + wb * torch.cat(slice_dict_b,
                                                                                     dim=1)  # [B, 97, H/2, W/2]
            slice = self.softmax(final_cost_volume)  # [B, 97, H/2, W/2]
            disparity_samples = torch.arange(0, 97, dtype=slice.dtype, device=slice.device).view(1, 97, 1,
                                                                                                 1)  # [1, 97, 1, 1]
            disparity_samples = disparity_samples.repeat(slice.size()[0], 1, slice.size()[2],
                                                         slice.size()[3])  # [B, 97, H/2, W/2]
            half_disp = torch.sum(disparity_samples * slice, dim=1).unsqueeze(1)  # [B, 1, H/2, W/2]
            left_half = F.interpolate(
                left_input,
                scale_factor=1 / pow(2, 1),
                mode='bilinear',
                align_corners=False)
            right_half = F.interpolate(
                right_input,
                scale_factor=1 / pow(2, 1),
                mode='bilinear',
                align_corners=False)
            refinement_disp = self.refinement_net(half_disp, left_half, right_half)
            out2 = F.interpolate(refinement_disp * 2.0, scale_factor=(2.0, 2.0),
                                 mode='bilinear', align_corners=False).squeeze(1)  # [B, H, W]
            return [out2]
