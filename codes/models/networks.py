import torch
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.FSRCNN_arch as FSRCNN_arch
import models.archs.CARN_arch as CARN_arch
import models.archs.carn_arch as carn_arch
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'SRResNet':
        netG = SRResNet_arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'SRResNet_noBN':
        netG = SRResNet_arch.SRResNet_noBN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'SRResNet_up':
        netG = SRResNet_arch.SRResNet_BN_up(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'SRResNet_MHCA_up':
        netG = SRResNet_arch.SRResNet_MHCA_up(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'SRResNet_noBN_up':
        netG = SRResNet_arch.SRResNet_noBN_up(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'SRResNet_MHCA':
        netG = SRResNet_arch.SRResNet_MHCA(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'SRResNet_ECA':
        netG = SRResNet_arch.SRResNet_ECA(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'SRResNet_CSAM':
        netG = SRResNet_arch.SRResNet_CSAM(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'SRResNet_MHCA':
        netG = SRResNet_arch.SRResNet_MHCA(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'SRResNet_noBN_MHCA_up':
        netG = SRResNet_arch.SRResNet_noBN_MHCA_up(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                               nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'CARN_M':
        netG = CARN_arch.CARN_M(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                nf=opt_net['nf'], scale=opt_net['scale'], group=opt_net['group'])
    elif which_model == 'CARN':
        netG = carn_arch.Net(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                             nf=opt_net['nf'], scale=opt_net['scale'], group=opt_net['group'])
    elif which_model == 'CARN_eca':
        netG = carn_arch.Net_eca(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                 nf=opt_net['nf'], scale=opt_net['scale'], group=opt_net['group'])
    elif which_model == 'CARN_MHCA':
        netG = CARN_arch.CARN_MHCA(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                   nf=opt_net['nf'], scale=opt_net['scale'], group=opt_net['group'])
    elif which_model == 'fsrcnn':
        netG = FSRCNN_arch.FSRCNN_net(input_channels=opt_net['in_nc'], upscale=opt_net['scale'], d=opt_net['d'],
                                      s=opt_net['s'], m=opt_net['m'])
    elif which_model == 'fsrcnn-MHCA':
        netG = FSRCNN_arch.FSRCNN_MHCA(input_channels=opt_net['in_nc'], upscale=opt_net['scale'], d=opt_net['d'],
                                       s=opt_net['s'], m=opt_net['m'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
