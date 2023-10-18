import torch
import torchvision
import torch.nn.functional as F

from .backbone import *
from .drn import *
from .criterion import BCELoss, L1Loss, L2Loss, BinaryLoss
from .transformer import SwinTransformer
import argparse

def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'no':
        return x
    else:
        raise Exception('Unkown activation!')

class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)
        
    def build_sound_ground(self, arch='unet6', weights=''):
        # 2D models
        if arch == 'anet':
            net_sound = ANet()
        elif arch == 'unet':
            net_sound = Unet(fc_dim=64, num_downs=7)
        elif arch == 'vggish':
            net_sound = VGGish()
        else:
            raise Exception('Architecture undefined!')
        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))

        return net_sound

    # builder for vision
    def build_frame_ground(self, arch='resnet18', pool_type='maxpool',
                    weights=''):
        pretrained=True
        if arch == 'resnet18':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = Resnet(
                original_resnet, pool_type=pool_type)
        elif arch == 'swintransformer':
            net = SwinTransformer()
        elif arch == 'drn':
            net = drn_c_26(pretrained=pretrained)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights))
        return net

    def build_tnet(self, args, arch, weights=''):

        if arch == 'base':
            net = TemporalNet()
        else:
            raise Exception('Architecture undefined!')

        # net.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for temporal net')
            net.load_state_dict(torch.load(weights))
        return net

    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'l1':
            net = L1Loss()
        elif arch == 'l2':
            net = L2Loss()
        elif arch == 'bn':
            net = BinaryLoss()
        else:
            raise Exception('Architecture undefined!')
        return net