import os
import random
import time
import sys
from pathlib import Path
import glob


# Numerical libs
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as tF
from sklearn.metrics import auc

# Image/audio process
from PIL import Image
from moviepy.editor import AudioFileClip, ImageSequenceClip
import cv2
import librosa
import torchaudio
import soundfile as sf

# Our libs
from arguments import ArgParser
from datasets.dataloader import EpicDataset
from network import ModelBuilder
from utils.main_utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
torch.autograd.set_detect_anomaly(True)


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, crit):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame, self.net_temporal = nets
        self.crit = crit
        self.m = nn.Sigmoid()
        self.epsilon = 0.5
        self.tau = 0.03
        self.cts = nn.CrossEntropyLoss(label_smoothing=0.).cuda()

    def forward(self, batch_data, args, attn_pos=None):
        img, spec, ori, spec_all = batch_data
  
        B, C, T, H, W = img.shape
        
        if attn_pos is None:
            attn_pos = T // 2 # Use the middle frame as the query frame for aggregation

        # audio preprocessing
        spec = spec + 1e-10
        
        # warp the spectrogram
        grid_warp = torch.from_numpy(warpgrid(spec.size(0), 128, 128, warp=True)).cuda()
        spec = F.grid_sample(spec, grid_warp)
        spec_clean = F.grid_sample(spec_all[:, 0, :, :, :], grid_warp)

        # calculate ground-truth masks
        gt_masks = (spec_clean > 0.5 * spec).float()

        # ------------- visual ----------------
        # extract image feat
        img_feat = self.net_frame.forward_multiframe(img, pool=False) # [B,C,T,H,W]

        # ------------- audio ----------------
        # extract audio feat
        sound_feat, pred_mask = self.net_sound.forward_with_separation(spec, img_feat)

        # ------------- temporal ----------------
        img_feat_attn, _ = self.net_temporal(attn_pos, img_feat, img, sound_feat)

        # normalization
        if img_feat_attn.shape[0] == 1:
            img_feat_attn = F.normalize(img_feat_attn[0], p=2, dim=0).unsqueeze(0)
            sound_feat = F.normalize(sound_feat[0], p=2, dim=0).unsqueeze(0)
        else:
            img_feat_attn = F.normalize(img_feat_attn, p=2, dim=1)
            sound_feat = F.normalize(sound_feat, p=2, dim=1)

        # Heatmap
        img_feat_attn = img_feat_attn.transpose(1,2).contiguous().view(-1, 512, 28, 28) # [B*T, C, H, W]
        sound_feat = (sound_feat.unsqueeze(1).repeat(1, T, 1).view(-1, img_feat_attn.shape[1]))
        A = torch.einsum('ncqa,nchw->nqa', [img_feat_attn, sound_feat.unsqueeze(2).unsqueeze(3)]).unsqueeze(1) # [B*T, 1, H, W]
        A0 = torch.einsum('ncqa,ckhw->nkqa', [img_feat_attn, sound_feat.T.unsqueeze(2).unsqueeze(3)]) # [B*T, B*T, H, W]

        # heatmap to return
        A_return = A.view(B, T, 1, 28, 28)[:, attn_pos, ...]

        # MIL with pooling function
        A_bag = A.view(B, T, 1, 28, 28)
        A_all = A_bag
        w_AT = F.softmax(A_bag, dim=1)
        A_bag = (A_bag*w_AT).sum(dim=1)
        A_bag = A_bag.repeat(1, T, 1, 1, 1)
        A_bag = A_bag.view(-1, 1, 28, 28)

        # positive and negative signals 
        Pos_mask = self.m((A - self.epsilon)/self.tau) 
        Pos_bag_mask = self.m((A_bag - self.epsilon)/self.tau) 
        Neg_mask = self.m((A0 - self.epsilon)/self.tau) 

        Pos = (Pos_mask * A).view(*A.shape[:2],-1).sum(-1) / (Pos_mask.view(*Pos_mask.shape[:2],-1).sum(-1))
        Pos_bag = (Pos_bag_mask * A_bag).view(*A_bag.shape[:2],-1).sum(-1) / (Pos_bag_mask.view(*Pos_bag_mask.shape[:2],-1).sum(-1))
        zero_mask = torch.zeros(A0.shape[0], A0.shape[0]).cuda()
        ones_mask = torch.ones(T, T).cuda()
        for i in range(B):
            zero_mask[i*T:(i+1)*T, i*T:(i+1)*T] = ones_mask
        optim_mask = 1 - 1 * zero_mask
        Neg = (Neg_mask * A0).view(*A0.shape[:2],-1).sum(-1) / (Neg_mask.view(*Neg_mask.shape[:2],-1).sum(-1)) * optim_mask

        # loss calculation
        logits = torch.cat((Pos, Neg), dim=1) / 0.1
        logits_bag = torch.cat((Pos_bag, Neg), dim = 1) / 0.1

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.cts(logits_bag, labels)

        loss = loss + 10 * F.mse_loss(pred_mask, gt_masks.detach()) 
        # loss = loss + F.binary_cross_entropy(pred_mask, gt_masks.detach()) # For binary mask, you can also try binary cross entropy loss

        return loss, A_return, A_all

# train one epoch
def train(netWrapper, loader, optimizer, history, epoch, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        # forward pass
        netWrapper.zero_grad()
        err, _, _ = netWrapper.forward(batch_data, args, attn_pos=None)
        err = err.mean()

        # backward
        err.backward()
        optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        torch.cuda.empty_cache() 
        
        # display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_frame: {}, '
                  'loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_frame, 
                          err.item()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.item())


def checkpoint(nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_sound, net_frame, net_temporal) = nets
    suffix_latest = '%d.pth'%epoch

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_sound.state_dict(),
               '{}/sound_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_frame.state_dict(),
               '{}/frame_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_temporal.state_dict(),
               '{}/tnet_{}'.format(args.ckpt, suffix_latest))

def create_optimizer(nets, args):
    (net_sound, net_frame, net_temporal) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound},
                    {'params': net_frame.parameters(), 'lr': args.lr_frame},
                    {'params': net_temporal.parameters(), 'lr': args.lr_frame}]
    return torch.optim.Adam(param_groups)

def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_sound = builder.build_sound_ground(
        arch=args.arch_sound,
        weights=args.weights_sound)
    net_frame = builder.build_frame_ground(
        arch=args.arch_frame,
        pool_type=args.img_pool,
        weights=args.weights_frame)
    net_temporal = builder.build_tnet(
        args=args,
        arch='base',
        weights=args.weights_tnet)
    nets = (net_sound, net_frame, net_temporal)
    crit = builder.build_criterion(arch=args.loss)

    # Dataset and Loader
    dataset_train = EpicDataset(
        args.list_train, max_sample=-1, args=args, split='train')
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)

    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # Wrap networks
    netWrapper = NetWrapper(nets, crit)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)

    # Set up optimizer
    optimizer = create_optimizer(nets, args)

    # History of performance
    history = {
        'train': {'epoch': [], 'err': []}}

    # Training loop
    for epoch in range(1, args.num_epoch + 1):
        train(netWrapper, loader_train_epic, optimizer, history, epoch, args)


        # Checkpointing
        if epoch % args.eval_epoch == 0:
            checkpoint(nets, history, epoch, args)
        
        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")

    # experiment name
    if args.log_freq:
        args.id += '-LogFreq'
    args.id += '-{}-{}'.format(
        args.arch_frame, args.arch_sound)
    args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
    args.id += '-channels{}'.format(args.num_channels)
    args.id += '-epoch{}'.format(args.num_epoch)
    args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])

    print('Model ID: {}'.format(args.id))
    
    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')

    if args.mode == 'train':
        makedirs(args.ckpt, remove=False)
        makedirs(args.vis, remove=False)
        args.weights_frame = ''
        args.weights_sound = ''
        args.weights_tnet = ''
    elif args.mode == 'eval':
        args.weights_frame = os.path.join(args.ckpt, args.weights_frame)
        args.weights_sound = os.path.join(args.ckpt, args.weights_sound)
        args.weights_tnet = os.path.join(args.ckpt, args.weights_tnet)

    # initialize best error with a big number
    args.best_err = inf

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)