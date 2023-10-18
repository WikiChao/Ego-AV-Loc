from sklearn.preprocessing import KernelCenterer
import torch
import torch.nn.functional as F
import torch.nn as nn
from .network_blocks import *
import torchgeometry as tgm
from utils.main_utils import *


class R2Plus1D(nn.Module):
    def __init__(self, depth=18):
        super(R2Plus1D, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        if depth == 10:
            self.conv2x = BasicR2P1DBlock(64, 64)
            self.conv3x = BasicR2P1DBlock(64, 128, stride=(2, 2, 2))
            self.conv4x = BasicR2P1DBlock(128, 256, stride=(2, 2, 2))
            self.conv5x = BasicR2P1DBlock(256, 512, stride=(2, 2, 2))
        elif depth == 18:
            self.conv2x = nn.Sequential(BasicR2P1DBlock(64, 64), BasicR2P1DBlock(64, 64))
            self.conv3x = nn.Sequential(BasicR2P1DBlock(64, 128, stride=(2, 2, 2)), BasicR2P1DBlock(128, 128))
            self.conv4x = nn.Sequential(BasicR2P1DBlock(128, 256, stride=(2, 2, 2)), BasicR2P1DBlock(256, 256))
            self.conv5x = nn.Sequential(BasicR2P1DBlock(256, 512, stride=(2, 2, 2)), BasicR2P1DBlock(512, 512))

        self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.out_dim = 512

    def forward(self, x, return_embs=False):
        x_c1 = self.conv1(x)
        x_b1 = self.conv2x(x_c1)
        x_b2 = self.conv3x(x_b1)
        x_b3 = self.conv4x(x_b2)
        x_b4 = self.conv5x(x_b3)
        x_pool = self.pool(x_b4)
        if return_embs:
            return {'conv1': x_c1, 'conv2x': x_b1, 'conv3x': x_b2, 'conv4x': x_b3, 'conv5x': x_b4, 'pool': x_pool}
        else:
            return x_pool

    def FLOPs(self, inpt_size):
        import numpy as np
        size = self.conv1(torch.randn(inpt_size, device=self.conv1[0].weight.device)).shape
        flops = np.prod(self.conv1[0].weight.shape) * np.prod(size[2:])

        for convx in [self.conv2x, self.conv3x, self.conv4x, self.conv5x]:
            for mdl in convx:
                flops_tmp, size = mdl.FLOPs(size)
                flops += flops_tmp

        return flops, size


class VGGish3D(nn.Module):
    def __init__(self, depth=10):
        super(VGGish3D, self).__init__()
        assert depth == 10

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=7, padding=3, stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.block1 = Basic3DBlock(64, 64)
        self.block2 = Basic3DBlock(64, 128, stride=(2, 2, 2))
        self.block3 = Basic3DBlock(128, 256, stride=(2, 2, 2))
        self.block4 = Basic3DBlock(256, 512)

        self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.out_dim = 512

    def forward(self, x, return_embs=False):
        x_c1 = self.conv1(x)
        x_b1 = self.block1(x_c1)
        x_b2 = self.block2(x_b1)
        x_b3 = self.block3(x_b2)
        x_b4 = self.block4(x_b3)
        x_pool = self.pool(x_b4)

        if return_embs:
            return {'conv1': x_c1, 'block1': x_b1, 'block2': x_b2, 'block3': x_b3, 'block4': x_b4, 'pool': x_pool}
        else:
            return x_pool

    def FLOPs(self, inpt_size):
        import numpy as np
        size = self.conv1(torch.randn(inpt_size)).shape
        flops = np.prod(self.conv1[0].weight.shape) * np.prod(size[2:])

        flops_tmp, size = self.block1.FLOPs(size)
        flops += flops_tmp

        flops_tmp, size = self.block2.FLOPs(size)
        flops += flops_tmp

        flops_tmp, size = self.block3.FLOPs(size)
        flops += flops_tmp

        flops_tmp, size = self.block4.FLOPs(size)
        flops += flops_tmp
        return flops, size

class Conv2D(nn.Module):
    def __init__(self, depth=10, inp_channels=1):
        super(Conv2D, self).__init__()
        assert depth==10

        self.conv1 = nn.Sequential(
            nn.Conv2d(inp_channels, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.block1 = Basic2DBlock(64, 64)
        self.block2 = Basic2DBlock(64, 128, stride=(2, 2))
        self.block3 = Basic2DBlock(128, 256, stride=(2, 2))
        self.block4 = Basic2DBlock(256, 512)

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.out_dim = 512

    def forward(self, x, return_embs=False):
        x_c1 = self.conv1(x)
        x_b1 = self.block1(x_c1)
        x_b2 = self.block2(x_b1)
        x_b3 = self.block3(x_b2)
        x_b4 = self.block4(x_b3)
        x_pool = self.pool(x_b4)

        if return_embs:
            return {'conv1': x_c1, 'block1': x_b1, 'block2': x_b2, 'block3': x_b3, 'block4': x_b4, 'pool': x_pool}
        else:
            return x_pool

    def FLOPs(self, inpt_size):
        import numpy as np
        size = self.conv1(torch.randn(inpt_size)).shape
        flops = np.prod(self.conv1[0].weight.shape) * np.prod(size[2:])

        flops_tmp, size = self.block1.FLOPs(size)
        flops += flops_tmp

        flops_tmp, size = self.block2.FLOPs(size)
        flops += flops_tmp

        flops_tmp, size = self.block3.FLOPs(size)
        flops += flops_tmp

        flops_tmp, size = self.block4.FLOPs(size)
        flops += flops_tmp
        return flops, size

class ANet(nn.Module):
    def __init__(self, init_weights=True):
        super(ANet, self).__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(1,  64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        b = x.size(0)
        x = F.adaptive_max_pool2d(x, 1).view(b, 512)
        return x

class VGGish(nn.Module):
    def __init__(self, init_weights=True):
        super(VGGish, self).__init__()
        self.unet = AudioVisual5layerUNet(ngf=64, input_nc=1, output_nc=1)
        
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(1,  64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    #    if init_weights:
    #        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward_with_separation(self, spec, visual_feat):
        x = torch.log(spec).detach()
        visual_feat = torch.mean(visual_feat, dim=2) # [B, C, T, H, W] -> [B, C, H, W]
        visual_feat = F.adaptive_max_pool2d(visual_feat, 1)
        mask, mask_feat = self.unet(x, visual_feat)
        b = x.size(0)
        x_mask = F.adaptive_max_pool2d(mask_feat, 1).view(b, 512)
        return x_mask, mask

    def forward(self, x):
        x = torch.log(x).detach()
        x = self.features(x)
        b = x.size(0)
        x = F.adaptive_max_pool2d(x, 1).view(b, 512)
        return x

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))

    if(Relu):
        model.append(nn.ReLU())

    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class AudioVisual7layerUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisual7layerUNet, self).__init__()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = unet_conv(ngf * 8, ngf * 8)

        self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

    def forward(self, x, visual_feat):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)

        visual_feat = visual_feat.repeat(1, 1, audio_conv7feature.shape[2], audio_conv7feature.shape[3])
        audioVisual_feature = torch.cat((visual_feat, audio_conv7feature), dim=1)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv6feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv5feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv4feature), dim=1))
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv3feature), dim=1))
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv1feature), dim=1))
        return mask_prediction

class AudioVisual5layerUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisual5layerUNet, self).__init__()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

        self.linear = nn.Sequential(
            nn.Conv2d(ngf * 16, ngf * 8, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, 1)
        )

    def forward(self, x, visual_feat):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[2], audio_conv5feature.shape[3])
        audioVisual_feature = torch.cat((visual_feat, audio_conv5feature), dim=1)
        audioVisual_feature = self.linear(audioVisual_feature)

        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        return mask_prediction, audioVisual_feature

class Unet(nn.Module):
    def __init__(self, fc_dim=64, num_downs=5, ngf=64, use_dropout=False, init_weights=True):
        super(Unet, self).__init__()
        
        # construct unet structure
        unet_block = UnetBlock(
            ngf * 8, ngf * 8, input_nc=None,
            submodule=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(
                ngf * 8, ngf * 8, input_nc=None,
                submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetBlock(
            ngf * 4, ngf * 8, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            ngf * 2, ngf * 4, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            ngf, ngf * 2, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            fc_dim, ngf, input_nc=1,
            submodule=unet_block, outermost=True)

        self.bn0 = nn.BatchNorm2d(1)
        self.unet_block = unet_block

        self.linear = nn.Sequential(
            nn.Conv2d(fc_dim, 128, 1),
            nn.Dropout(0.),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1)
        )

        self.inner_prod = InnerProd(fc_dim=512)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_with_separation(self, spec, visual_feat):
        B, C, H, W = spec.shape
        x = torch.log(spec).detach()
        visual_feat = torch.mean(visual_feat, dim=2) # [B, C, T, H, W] -> [B, C, H, W]
        visual_feat = F.adaptive_max_pool2d(visual_feat, 1)
        spec_feat = self.forward(x)
        spec_feat = self.linear(spec_feat)
        mask = self.inner_prod(visual_feat, spec_feat)
        mask = F.sigmoid(mask)
        spec_feat = mask * spec_feat
        spec_feat = F.adaptive_max_pool2d(spec_feat, 1).view(B, 512)
        return spec_feat, mask

    def forward(self, x):
        x = self.bn0(x)
        x = self.unet_block(x)

        return x

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_input_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 use_dropout=False, inner_output_nc=None, noskip=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.noskip = noskip
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if innermost:
            inner_output_nc = inner_input_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_input_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        if outermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1)

            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)

            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost or self.noskip:
            return self.model(x)
        else:
            #print(x.size(), self.model(x).size())
            return torch.cat([x, self.model(x)], 1)

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Grounding(nn.Module):
    def __init__(self):
        super(Grounding, self).__init__()

        # visual net dimension reduction
        self.fc_v1 = nn.Linear(512, 128)
        self.fc_v2 = nn.Linear(128, 128)

        # audio net dimension reduction
        self.fc_a1 = nn.Linear(512, 128)
        self.fc_a2 = nn.Linear(128, 128)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

        self.relu = nn.ReLU()
        self.bn = LBSign.apply
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, feat_sound, feat_img):

        feat = torch.cat((feat_sound,  feat_img), dim =-1)
        g = self.fc3(self.relu(self.fc2(self.relu(self.fc1(feat)))))
        g = g*2
        return g

class Resnet(nn.Module):
    def __init__(self, original_resnet,pool_type='maxpool'):
        super(Resnet, self).__init__()
        self.pool_type = pool_type
        self.features = nn.Sequential(
            *list(original_resnet.children())[:-2])
        # for param in self.features.parameters():
        #     param.requires_grad = False

    def forward(self, x, pool=True):
        x = self.features(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        x = self.features(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)

        x = x.view(B, C)
        return x

class TemporalNet(nn.Module):
    def __init__(self, num_dims=512, num_heads=1, dropout=0.0, tau=5e1, scale_factor=1e-1):
        super(TemporalNet, self).__init__()
        
        # define temproal modeling network
        self.spa_attn = nn.Sequential(
            nn.Conv2d(num_dims*2, num_dims, 1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_dims, num_dims, 1)
        )
        self.temp_attn = nn.MultiheadAttention(num_dims, num_heads)
        self.norm = nn.LayerNorm(num_dims)
        self.dropout = nn.Dropout(dropout)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(num_dims, num_dims),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(num_dims, num_dims)
        )

        self.tau = tau
        self.scale_factor = scale_factor

    def transform_view(self, K, K_inv, RTinv_cam1, RT_cam2):
        # Transform into camera coordinate of the first view
        cam1_X = K_inv.bmm(xys)

        # Transform into world coordinates
        RT = RT_cam2.bmm(RTinv_cam1)
        wrld_X = RT.bmm(cam1_X)

        # And intrinsics
        xy_proj = K.bmm(wrld_X)
        return xy_proj

    def noramlize(self, batch):
        B, C, H, W = batch.size()
        batch = batch.contiguous().view(batch.size(0), -1)
        batch -= batch.min(1, keepdim=True)[0]
        batch /= batch.max(1, keepdim=True)[0]
        batch = batch.view(B, C, H, W)
        return batch
    
    def homo_normalize(self, homo):
        homo = homo.view(-1, 1)
        sum = [homo[i][0] ** 2 for i in range(homo.shape[0])]
        scale_factor = torch.sqrt(torch.sum(torch.stack(sum)))
        homo[:-2] = homo[:-2] / scale_factor 
        return homo.view(3, 3)

    def homo_transform(self, frames, attn_pos):
        B, C, T, H, W = frames.size()
        WARPED_IMG = []
        MASK = []
        MASK_inv = []
        for b in range(B):
            # for each batch do normalization
            batch_frames = frames[b, :, :, :, :].clone().transpose(0, 1) # [C, T, H, W]
            masks = torch.ones_like(batch_frames).cuda()
            query_img = batch_frames[attn_pos, :, :, :] 
            query_img = self.noramlize(query_img.unsqueeze(0)) * 255.
            Homo = [] # store homography matrix
            Homo_inv = [] # store homoinv

            for t in range(T):
                if t == attn_pos:
                    Homo.append(torch.eye(3))
                    Homo_inv.append(torch.eye(3))
                    continue
                tmp_img = batch_frames[t, :, :, :] 
                tmp_img = self.noramlize(tmp_img.unsqueeze(0)) * 255.
                h = homo(tensor2cv(tmp_img[0]).astype(np.uint8), tensor2cv(query_img[0]).astype(np.uint8))
                Homo_inv.append(h)
                h = torch.linalg.inv(h)
                Homo.append(h)

            Homo = torch.stack(Homo).type(torch.FloatTensor).cuda() # [T,3,3]
            Homo_inv = torch.stack(Homo_inv).type(torch.FloatTensor).cuda() # [T,3,3]
            
            warper = tgm.HomographyWarper(batch_frames.shape[2], batch_frames.shape[3], normalized_coordinates=False)
            grid = warper.warp_grid(Homo)
            grid_inv = warper.warp_grid(Homo_inv)
            warped_img = F.grid_sample(batch_frames, (grid/grid.shape[-2] - 0.5)*2, align_corners = True) 
            warped_mask = F.grid_sample(masks, (grid/grid.shape[-2] - 0.5)*2, align_corners = True)
            warped_mask_inv = F.grid_sample(warped_mask, (grid_inv/grid_inv.shape[-2] - 0.5)*2, align_corners = True)
            WARPED_IMG.append(warped_img.transpose(0,1))
            MASK.append(warped_mask.transpose(0,1))
            MASK_inv.append(warped_mask_inv.transpose(0,1))

        WARPED_IMG = torch.stack(WARPED_IMG) # [B,C,T,H,W]
        MASK = torch.stack(MASK) # [B,C,T,H,W]
        MASK_inv = torch.stack(MASK_inv) # [B,C,T,H,W]
        return WARPED_IMG, MASK, MASK_inv

    def temporal_aggregation(self, feat, attn_pos, mask=None):
        """
        Aggregate the visual feature along the time axis
        feat: [B, C, T, H, W] 
        attn_pos: [B, C, T, H, W] 
        """
        B, C, T, H, W = feat.size()
        
        feat_tem = feat.permute(2, 0, 3, 4, 1).contiguous().view(T, -1, C)
        tem_out = self.temp_attn(feat_tem, feat_tem, feat_tem)[0]
        feat_out = feat_tem + self.dropout(tem_out) # [T, B*H*W, C]
        linear_out = self.linear_net(feat_out)
        feat_out = feat_out +  self.dropout(linear_out)
        feat_out = feat_out.view(T, B, H, W, C).permute(1, 4, 0, 2, 3)
        
        return feat_out

    def warp_feature(self, attn_pos, frames_feat, frames, scale_factor = 8.):
        """
        warp an image/tensor (im2) back to im1, according to the homography matrix
        attn_pos: the temporal position of the key frame
        frames_feat: [B, C, T, H, W] 
        frames: [B, C, T, H, W] 
        """
        B, C, T, H, W = frames_feat.size()
        WARPED_FEAT = []
        
        S = torch.eye(3)
        S[0,0] *= 1 / scale_factor
        S[1,1] *= 1 / scale_factor

        normalization_matrix = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], device=S.device, dtype=S.dtype)
        width_denom = frames.shape[-1] / scale_factor - 1.0
        height_denom = frames.shape[-2] / scale_factor - 1.0
        normalization_matrix[0, 0] = normalization_matrix[0, 0] * 2.0 / width_denom
        normalization_matrix[1, 1] = normalization_matrix[1, 1] * 2.0 / height_denom

        # global transformation
        for b in range(B):
            # for each batch do normalization
            batch_frames = frames[b, :, :, :, :].clone().transpose(0, 1) # [T, C, H, W]
            query_img = batch_frames[attn_pos, :, :, :] 
            query_img = self.noramlize(query_img.unsqueeze(0)) * 255.
            Homo = [] # store homography matrix
            tmp_feat = frames_feat[b, :, :, :, :].clone()

            for t in range(T):
                if t == attn_pos:
                    Homo.append(torch.eye(3))
                    continue
                tmp_img = batch_frames[t, :, :, :] 
                tmp_img = self.noramlize(tmp_img.unsqueeze(0)) * 255.
                h = homo(tensor2cv(tmp_img[0]).astype(np.uint8), tensor2cv(query_img[0]).astype(np.uint8)).float()
                
                # downscaling homography matrix
                h = torch.matmul(S, torch.matmul(h, torch.inverse(S)))
                
                # normalize to [-1,1] for F.grid_sample
                h = torch.matmul(normalization_matrix, torch.matmul(h, torch.inverse(normalization_matrix)))

                h = torch.linalg.inv(h)
                Homo.append(h)
            
            Homo = torch.stack(Homo).type(torch.FloatTensor).cuda() # [T,3,3]
            warper = tgm.HomographyWarper(tmp_feat.shape[2], tmp_feat.shape[3], normalized_coordinates=True)
            # grid = warper.warp_grid(Homo)
            # warped_feat = F.grid_sample(tmp_feat.transpose(0,1), (grid/grid.shape[-2] - 0.5)*2, align_corners = True)
            warped_feat = warper(tmp_feat.transpose(0,1), Homo)
            WARPED_FEAT.append(warped_feat.transpose(0,1))

        WARPED_FEAT = torch.stack(WARPED_FEAT) # [B,C,T,H,W]

        return WARPED_FEAT


    def localization_mask(self, audio_feat, visual_feat):
        """
        calculate the soft localization mask on the intermidate visual feature maps, and reweight the visual feature to highlight the potential sounding region
        audio_feat: [B, C] 
        visual_feat: [B, C, T, H, W] 
        """
        B, C, T, H, W = visual_feat.size()

        # normalization
        if visual_feat.shape[0] == 1:
            visual_feat = F.normalize(visual_feat[0], p=2, dim=0).unsqueeze(0)
            audio_feat = F.normalize(audio_feat[0], p=2, dim=0).unsqueeze(0)
        else:
            visual_feat = F.normalize(visual_feat, p=2, dim=1)
            audio_feat = F.normalize(audio_feat, p=2, dim=1)

        # sounding object probability
        visual_feat_ = visual_feat.transpose(1,2).contiguous().view(-1, C, H, W) # [B*T, C, H, W]
        audio_feat_ = (audio_feat.unsqueeze(1).repeat(1, T, 1).view(-1, C))

        prob = torch.einsum('ncqa,nchw->nqa', [visual_feat_, audio_feat_.unsqueeze(2).unsqueeze(3)]).unsqueeze(1) # [B*T, 1, H, W]
        prob = prob.view(-1, 1, H*W)
        weights = F.softmax(prob / self.tau, dim=-1) 
        weights = weights.view(B, T, 1, H, W).transpose(1, 2)
        visual_feat = self.scale_factor * visual_feat * weights + visual_feat

        return visual_feat, weights

    def forward(self, attn_pos, frames_feat, frames, sound_feat):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        key_pos: the temporal position of the key frame
        frames_feat: [B, C, T, H, W] 
        frames: [B, C, T, H, W] 
        """
        B, C, T, H, W = frames_feat.size()

        frames_feat_out = frames_feat.clone()
        WARPED_FEAT = []
       
        # # applied localization mask (The training might be unstable with adding this module, we recommend to add this module during the fine-tune stage.)
        # frames_feat, weights = self.localization_mask(sound_feat, frames_feat)

        # homography estimation based on query frame (usually choosed as the middle frame)
        warp_feat_tmp = self.warp_feature(attn_pos=attn_pos, frames_feat=frames_feat, frames=frames)
        WARPED_FEAT.append(warp_feat_tmp)

        # temporal aggregation
        frames_feat_out = self.temporal_aggregation(warp_feat_tmp, attn_pos=attn_pos)

        # # You can treat each of the frames as keyframe to update frame features, just wrapped them in a for loop
        # for attn_pos in range(T):
        #     warp_feat_tmp = self.warp_feature(attn_pos=attn_pos, frames_feat=frames_feat, frames=frames)
        #     frames_feat_out[:,:,attn_pos,...] = self.temporal_aggregation(warp_feat_tmp, attn_pos=attn_pos)[:,:,attn_pos,...]

        return frames_feat_out, [frames_feat, WARPED_FEAT]

