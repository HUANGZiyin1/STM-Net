import torch
import torch.nn as nn


class STM(nn.Module):
    def __init__(self, nf=48, base_ks=3):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(STM, self).__init__()

        self.S_in = nn.Conv2d(1,nf,base_ks,padding=base_ks//2)
        self.ResL_in = nn.Conv2d(3,nf,base_ks,padding=base_ks//2)
        self.ResR_in = nn.Conv2d(3,nf,base_ks,padding=base_ks//2)
        self.MSRB1 = MSRB(nf)
        self.MSRB2 = MSRB(nf)
        self.MSRB3 = MSRB(nf)
        self.MSRB4 = MSRB(nf)
        self.ResL1 = ResBlock(nf)
        self.ResL2 = CrossBlock(nf)
        self.ResL3 = CrossBlock(nf)
        self.ResL4 = CrossBlock(nf)
        self.ResR1 = ResBlock(nf)
        self.ResR2 = CrossBlock(nf)
        self.ResR3 = CrossBlock(nf)
        self.ResR4 = CrossBlock(nf)

        self.rec= nn.Sequential(
            nn.Conv2d(nf*3, nf, base_ks, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf,1,base_ks,padding = 1)
        )



    def forward(self, inputs):
        TG = inputs[:,2, ...].unsqueeze(1)
        input_l = inputs[:,0:3, ...]
        input_r = inputs[:, 2:5, ...]
        o_s = self.MSRB4(self.MSRB3(self.MSRB2(self.MSRB1(self.S_in(TG)))))
        o_l1 = self.ResL1(self.ResL_in(input_l))
        o_r1 = self.ResR1(self.ResR_in(input_r))
        o_l2 = self.ResL2(torch.cat([o_l1,o_r1],1))
        o_r2 = self.ResR2(torch.cat([o_l1,o_r1],1))
        o_l3 = self.ResL3(torch.cat([o_l2,o_r2],1))
        o_r3 = self.ResR3(torch.cat([o_l2,o_r2],1))
        o_l4 = self.ResL4(torch.cat([o_l3,o_r3],1))
        o_r4 = self.ResR4(torch.cat([o_l3,o_r3],1))
        o = self.rec(torch.cat([o_s,o_l4,o_r4],1))
        o+=TG

        return o



class MSRB(nn.Module):
    def __init__(self, nf, k1 = 3, k2 = 5):
        super(MSRB, self).__init__()

        self.conv_5_1 = nn.Sequential(
            nn.Conv2d(nf, nf, k2, padding=k2//2),
            nn.ReLU(inplace=True),
        )

        self.conv_5_2 = nn.Sequential(
            nn.Conv2d(nf, nf, k2, padding=k2//2),
            nn.ReLU(inplace=True),
            CALayer(nf,16),
        )

        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(nf, nf, 1, padding=1//2),
            nn.ReLU(inplace=True),
        )

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(nf, nf, k1, padding=k1//2),
            nn.ReLU(inplace=True),
        )

        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(nf, nf, k1, padding=k1//2),
            nn.ReLU(inplace=True),
            CALayer(nf, 16),
        )

        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(nf, nf, 1, padding=1//2),
            nn.ReLU(inplace=True),
        )

        self.confusion = nn.Conv2d(nf * 3, nf, 1, padding=0, stride=1)

    def forward(self, x):
        input_1 = x
        output_5_1 = self.conv_5_1(input_1)
        output_5_2 = self.conv_5_2(input_1)
        output_1_1 = self.conv_1_1(input_1)
        output_1 = output_1_1 + output_5_1

        output_3_1 = self.conv_3_1(output_1)
        output_3_2 = self.conv_3_2(output_1)
        output_1_2 = self.conv_1_2(output_1)
        output_2 = output_1_2 + output_3_1

        input_3 = torch.cat([output_5_2, output_3_2, output_2], 1)
        output = self.confusion(input_3)
        output += x
        return output

class CrossBlock(nn.Module):
    def __init__(self, channel):
        super(CrossBlock, self).__init__()
        self.conv_bt = nn.Conv2d(channel*2, channel, 3, padding=3//2)
        self.Res = ResBlock(channel)

    def forward(self, x):
        o = self.Res(self.conv_bt(x))
        return o



class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.conv_bt = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=3 // 2),
        )

    def forward(self, x):
        res = self.conv_bt(x)
        res = res+x
        return res



class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y






if __name__ == '__main__':
    model = STM()
    net_input = torch.randn(32, 5, 128, 128)
    net_output = model(net_input)
    print(net_output.shape)