import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch


class FSRCNN_net(torch.nn.Module):
    def __init__(self, input_channels, upscale, d=64, s=12, m=4):
        super(FSRCNN_net, self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU())

        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.body_conv = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.tail_conv = nn.ConvTranspose2d(in_channels=d, out_channels=input_channels, kernel_size=9,
                                            stride=upscale, padding=3, output_padding=1)

        arch_util.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)

        fea = self.body_conv(fea)
        out = self.tail_conv(fea)

        return out

class FSRCNN_MHCA(torch.nn.Module):
    def __init__(self, input_channels, upscale, d=64, s=12, m=4):
        super(FSRCNN_MHCA, self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU())
        self.a1 = MHCA(n_feats=d, ratio=0.5)
        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.body_conv = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.tail_conv = nn.ConvTranspose2d(in_channels=d, out_channels=input_channels, kernel_size=9,
                                            stride=upscale, padding=3, output_padding=1)

        arch_util.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.a1(fea)
        fea = self.body_conv(fea)
        out = self.tail_conv(fea)

        return out

class MHCA(nn.Module):
    def __init__(self, n_feats, ratio):
        """
        MHCA spatial-channel attention module.
        :param n_feats: The number of filter of the input.
        :param ratio: Channel reduction ratio.
        """
        super(MHCA, self).__init__()

        out_channels = int(n_feats // ratio)

        head_1 = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=n_feats, kernel_size=1, padding=0, bias=True)
        ]

        kernel_size_sam = 3
        head_2 = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=kernel_size_sam, padding=0,
                      bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=kernel_size_sam, padding=0,
                               bias=True)
        ]

        kernel_size_sam_2 = 5
        head_3 = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=kernel_size_sam_2, padding=0,
                      bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=kernel_size_sam_2, padding=0,
                               bias=True)
        ]

        self.head_1 = nn.Sequential(*head_1)
        self.head_2 = nn.Sequential(*head_2)
        self.head_3 = nn.Sequential(*head_3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res_h1 = self.head_1(x)
        res_h2 = self.head_2(x)
        res_h3 = self.head_3(x)
        m_c = self.sigmoid(res_h1 + res_h2 + res_h3)
        res = x * m_c
        return res