import torch
from torchsummaryX import summary

import models.archs.SRResNet_arch as SRResNet_arch

model = SRResNet_arch.MSRResNet(in_nc=3, out_nc=3, nf=64, nb=16, upscale=4)

summary(model, torch.zeros((1, 3, 32, 32)))

# input LR x2, HR size is 720p
# summary(model, torch.zeros((1, 3, 640, 360)))

# input LR x3, HR size is 720p
# summary(model, torch.zeros((1, 3, 426, 240)))

# input LR x4, HR size is 720p
# summary(model, torch.zeros((1, 3, 320, 180)))
