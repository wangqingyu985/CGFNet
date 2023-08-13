from __future__ import print_function
from models.submodules import SubModule, convbn_3d_lrelu, convbn_transpose_3d


class HourGlass(SubModule):
    def __init__(self, inplanes=16):
        super(HourGlass, self).__init__()

        self.conv1 = convbn_3d_lrelu(in_planes=inplanes, out_planes=inplanes * 2, kernel_size=3, stride=2, pad=1)
        self.conv2 = convbn_3d_lrelu(in_planes=inplanes * 2, out_planes=inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv1_1 = convbn_3d_lrelu(inplanes * 2, inplanes * 4, kernel_size=3, stride=2, pad=1)
        self.conv2_1 = convbn_3d_lrelu(inplanes * 4, inplanes * 4, kernel_size=3, stride=1, pad=1)

        self.conv3 = convbn_3d_lrelu(inplanes * 4, inplanes * 8, kernel_size=3, stride=2, pad=1)
        self.conv4 = convbn_3d_lrelu(inplanes * 8, inplanes * 8, kernel_size=3, stride=1, pad=1)

        self.conv5 = convbn_transpose_3d(inplanes * 8, inplanes * 4, kernel_size=3, padding=1,
                                         output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False)
        self.conv6 = convbn_transpose_3d(inplanes * 4, inplanes * 2, kernel_size=3, padding=1,
                                         output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False)
        self.conv7 = convbn_transpose_3d(inplanes * 2, inplanes, kernel_size=3, padding=1,
                                         output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False)
        self.last_for_guidance = convbn_3d_lrelu(inplanes, 32, kernel_size=3, stride=1, pad=1)
        self.weight_init()


class CoeffsPredictor(HourGlass):
    def __init__(self, hourglass_inplanes=16):
        super(CoeffsPredictor, self).__init__(hourglass_inplanes)

    def forward(self, input):  # [B, 16, 25, H/8, W/8]
        output0 = self.conv1(input)  # [B, 32, 25, H/16, W/16]
        output0_a = self.conv2(output0) + output0  # [B, 32, 25, H/16, W/16]

        output0 = self.conv1_1(output0_a)  # [B, 64, 25, H/32, W/32]
        output0_c = self.conv2_1(output0) + output0  # [B, 64, 25, H/32, W/32]

        output0 = self.conv3(output0_c)  # [B, 128, 25, H/64, W/64]
        output0 = self.conv4(output0) + output0  # [B, 128, 25, H/64, W/64]

        output1 = self.conv5(output0) + output0_c  # [B, 64, 25, H/32, W/32]
        output1 = self.conv6(output1) + output0_a  # [B, 32, 25, H/16, W/16]
        output1 = self.conv7(output1)  # [B, 16, 25, H/8, W/8]
        coeffs = self.last_for_guidance(output1).permute(0, 2, 1, 3, 4).contiguous()  # [B, 25, 32, H/8, W/8]
        return coeffs  # [B, 25, 32, H/8, W/8]
