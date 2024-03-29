import os.path as osp
import sys
import torch
try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    import models.archs.SRResNet_arch as SRResNet_arch
except ImportError:
    pass

pretrained_net = torch.load('/media/ubuntu/DISK8T/超分辨代码/PAN-master/experiments/MSSResNet_MHCA_DIV2K_archived_230517-211541/models/455000_G.pth')
crt_model = SRResNet_arch.MSRResNet_MHCA(in_nc=3, out_nc=3, nf=64, nb=16, upscale=3)
crt_net = crt_model.state_dict()

for k, v in crt_net.items():
    if k in pretrained_net and 'upconv1' not in k:
        crt_net[k] = pretrained_net[k]
        print('replace ... ', k)

# x4 -> x3
crt_net['upconv1.weight'][0:256, :, :, :] = pretrained_net['upconv1.weight'] / 2
crt_net['upconv1.weight'][256:512, :, :, :] = pretrained_net['upconv1.weight'] / 2
crt_net['upconv1.weight'][512:576, :, :, :] = pretrained_net['upconv1.weight'][0:64, :, :, :] / 2
crt_net['upconv1.bias'][0:256] = pretrained_net['upconv1.bias'] / 2
crt_net['upconv1.bias'][256:512] = pretrained_net['upconv1.bias'] / 2
crt_net['upconv1.bias'][512:576] = pretrained_net['upconv1.bias'][0:64] / 2

torch.save(crt_net, '../../experiments/pretrained_models/MSRResNetx3_455.pth')
