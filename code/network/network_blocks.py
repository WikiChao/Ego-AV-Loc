import torch
import torch.nn as nn
import numpy as np


class Basic2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=(1, 1)):
        self.__dict__.update(locals())
        super(Basic2DBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

    def FLOPs(self, inpt_size):
        import numpy as np
        flops = 0
        x = torch.randn(inpt_size)

        x = self.relu(self.bn1(self.conv1(x)))
        flops += np.prod(self.conv1.weight.shape) * np.prod(x.shape[2:])
        x = self.relu(self.bn2(self.conv2(x)))
        flops += np.prod(self.conv2.weight.shape) * np.prod(x.shape[2:])
        return flops, x.shape


class Basic3DBlockSpatial(nn.Module):
    def __init__(self, in_planes, out_planes, stride=(1, 1, 1)):
        super(Basic3DBlockSpatial, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

    def FLOPs(self, inpt_size):
        import numpy as np
        flops = 0
        x = torch.randn(inpt_size)

        x = self.relu(self.bn1(self.conv1(x)))
        flops += np.prod(self.conv1.weight.shape) * np.prod(x.shape[2:])
        x = self.relu(self.bn2(self.conv2(x)))
        flops += np.prod(self.conv2.weight.shape) * np.prod(x.shape[2:])
        return flops, x.shape


class Basic3DResBlockSpatial(nn.Module):
    def __init__(self, in_planes, out_planes, stride=(1, 1, 1)):
        super(Basic3DResBlockSpatial, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        if in_planes != out_planes or any([s!=1 for s in stride]):
            self.res = True
            self.res_conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0), bias=False)
        else:
            self.res = False

    def forward(self, x):
        x_main = self.conv2(self.relu(self.bn1(self.conv1(x))))
        x_res = self.res_conv(x) if self.res else x
        x = self.relu(self.bn2(x_main + x_res))
        return x

    def FLOPs(self, inpt_size):
        flops = 0
        x = torch.randn(inpt_size)

        x = self.relu(self.bn1(self.conv1(x)))
        flops += np.prod(self.conv1.weight.shape) * np.prod(x.shape[2:])
        x = self.relu(self.bn2(self.conv2(x)))
        flops += np.prod(self.conv2.weight.shape) * np.prod(x.shape[2:])
        return flops, x.shape


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=(1, 1, 1)):
        super(Basic3DBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

    def FLOPs(self, inpt_size):
        flops = 0
        x = torch.randn(inpt_size)

        x = self.relu(self.bn1(self.conv1(x)))
        flops += np.prod(self.conv1.weight.shape) * np.prod(x.shape[2:])
        x = self.relu(self.bn2(self.conv2(x)))
        flops += np.prod(self.conv2.weight.shape) * np.prod(x.shape[2:])
        return flops, x.shape


class Basic3DResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=(1, 1, 1)):
        super(Basic3DResBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        if in_planes != out_planes or any([s!=1 for s in stride]):
            self.res = True
            self.res_conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0), bias=False)
        else:
            self.res = False

    def forward(self, x):
        x_main = self.conv2(self.relu(self.bn1(self.conv1(x))))
        x_res = self.res_conv(x) if self.res else x
        x = self.relu(self.bn2(x_main + x_res))
        return x

    def FLOPs(self, inpt_size):
        flops = 0
        x = torch.randn(inpt_size)

        x = self.relu(self.bn1(self.conv1(x)))
        flops += np.prod(self.conv1.weight.shape) * np.prod(x.shape[2:])
        x = self.relu(self.bn2(self.conv2(x)))
        flops += np.prod(self.conv2.weight.shape) * np.prod(x.shape[2:])
        return flops, x.shape


class BasicR2P1DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=(1, 1, 1)):
        super(BasicR2P1DBlock, self).__init__()
        spt_stride = (1, stride[1], stride[2])
        tmp_stride = (stride[0], 1, 1)
        self.spt_conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=spt_stride, padding=(0, 1, 1), bias=False)
        self.spt_bn1 = nn.BatchNorm3d(out_planes)
        self.tmp_conv1 = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=tmp_stride, padding=(1, 0, 0), bias=False)
        self.tmp_bn1 = nn.BatchNorm3d(out_planes)

        self.spt_conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.spt_bn2 = nn.BatchNorm3d(out_planes)
        self.tmp_conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.out_bn = nn.BatchNorm3d(out_planes)

        self.relu = nn.ReLU(inplace=True)

        if in_planes != out_planes or any([s!=1 for s in stride]):
            self.res = True
            self.res_conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0), bias=False)
        else:
            self.res = False

    def forward(self, x):
        x_main = self.tmp_conv1(self.relu(self.spt_bn1(self.spt_conv1(x))))
        x_main = self.relu(self.tmp_bn1(x_main))
        x_main = self.tmp_conv2(self.relu(self.spt_bn2(self.spt_conv2(x_main))))

        x_res = self.res_conv(x) if self.res else x
        x_out = self.relu(self.out_bn(x_main + x_res))
        return x_out

    def FLOPs(self, inpt_size):
        flops = 0
        x = torch.randn(inpt_size, device=self.spt_conv1.weight.device)

        x = self.relu(self.spt_bn1(self.spt_conv1(x)))
        flops += np.prod(self.spt_conv1.weight.shape) * np.prod(x.shape[2:])
        x = self.relu(self.tmp_bn1(self.tmp_conv1(x)))
        flops += np.prod(self.tmp_conv1.weight.shape) * np.prod(x.shape[2:])
        x = self.relu(self.spt_bn2(self.spt_conv2(x)))
        flops += np.prod(self.spt_conv2.weight.shape) * np.prod(x.shape[2:])
        x = self.tmp_conv2(x)
        flops += np.prod(self.tmp_conv2.weight.shape) * np.prod(x.shape[2:])
        if self.res:
            x = torch.randn(inpt_size, device=self.spt_conv1.weight.device)
            x = self.relu(self.out_bn(self.res_conv(x)))
            flops += np.prod(self.res_conv.weight.shape) * np.prod(x.shape[2:])


        return flops, x.shape


class InnerProd(nn.Module):
    def __init__(self, fc_dim):
        super(InnerProd, self).__init__()
        self.scale = nn.Parameter(torch.ones(fc_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feat_img, feat_sound):
        sound_size = feat_sound.size()
        B, C = sound_size[0], sound_size[1]
        feat_img = feat_img.view(B, 1, C)
        z = torch.bmm(feat_img * self.scale, feat_sound.view(B, C, -1)) \
            .view(B, 1, *sound_size[2:])
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        feat_img = feat_img.view(B, C)
        z = (feat_img * self.scale).view(B, C, 1, 1) * feat_sound
        z = z + self.bias
        return z

    # inference purposes
    def forward_pixelwise(self, feats_img, feat_sound):
        (B, C, HI, WI) = feats_img.size()
        (B, C, HS, WS) = feat_sound.size()
        feats_img = feats_img.view(B, C, HI*WI)
        feats_img = feats_img.transpose(1, 2)
        feat_sound = feat_sound.view(B, C, HS * WS)
        z = torch.bmm(feats_img * self.scale, feat_sound) \
            .view(B, HI, WI, HS, WS)
        z = z + self.bias
        return z