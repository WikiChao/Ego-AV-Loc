import os, sys
from collections import deque
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import datetime
import matplotlib.pyplot as plt
import librosa
import sklearn
import cv2
import random
import math
from bounding_box import bounding_box as bb
import subprocess as sp
from threading import Timer
from sklearn.metrics import auc, average_precision_score


def makedirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
            print('removed existing directory...')
        else:
            return
    os.makedirs(path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.
        else:
            return self.avg.tolist()

def initialize_distributed_backend(args, ngpus_per_node):
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.rank == -1:
        args.rank = 0
    return args

def distribute_model_to_cuda(models, args, batch_size, num_workers, ngpus_per_node):
    squeeze = False
    if not isinstance(models, list):
        models = [models]
        squeeze = True

    for i in range(len(models)):
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                models[i].cuda(args.gpu)
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i], device_ids=[args.gpu])
            else:
                models[i].cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i])
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            models[i] = models[i].cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            models[i] = torch.nn.DataParallel(models[i]).cuda()

    if squeeze:
        models = models[0]

    if args.distributed and args.gpu is not None:
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        batch_size = int(batch_size / ngpus_per_node)
        num_workers = int((num_workers + ngpus_per_node - 1) / ngpus_per_node)

    return models, args, batch_size, num_workers

def build_dataloaders(cfg, num_workers, distributed, logger):
    train_loader = build_dataloader(cfg, cfg['train'], num_workers, distributed)
    logger.add_line("\n"+"="*30+"   Train data   "+"="*30)
    logger.add_line(str(train_loader.dataset))
    test_loader = build_dataloader(cfg, cfg['test'], num_workers, distributed)
    logger.add_line("\n"+"="*30+"   Train data   "+"="*30)
    logger.add_line(str(train_loader.dataset))
    return train_loader, test_loader


def build_dataloader(db_cfg, split_cfg, num_workers, distributed):
    import torch.utils.data as data
    import torch.utils.data.distributed
    if db_cfg['name'] == 'yt360':
        db = build_360_dataset(db_cfg, split_cfg)
    else:
        db = build_video_dataset(db_cfg, split_cfg)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(db)
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(
        db,
        batch_size=db_cfg['batch_size'],
        shuffle=False,
        drop_last=split_cfg['drop_last'],
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler)

    return loader

def save_checkpoint(state, is_best, model_dir='.', filename=None):
    if filename is None:
        filename = '{}/checkpoint.pth.tar'.format(model_dir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/model_best.pth.tar'.format(model_dir))


class CheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.rank = rank
        self.best_metric = 0.

    def save(self, epoch, filename=None, eval_metric=0., **kwargs):
        if self.rank != 0:
            return

        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True

        state = {'epoch': epoch}
        for k in kwargs:
            state[k] = kwargs[k].state_dict()

        if filename is None:
            save_checkpoint(state=state, is_best=is_best, model_dir=self.checkpoint_dir)
        else:
            save_checkpoint(state=state, is_best=False, filename='{}/{}'.format(self.checkpoint_dir, filename))

    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def checkpoint_fn(self, last=False, best=False):
        assert best or last
        assert not (last and best)
        if last:
            return self.last_checkpoint_fn()
        if best:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def restore(self, fn=None, restore_last=False, restore_best=False, **kwargs):
        checkpoint_fn = fn if fn is not None else self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        start_epoch = ckp['epoch']
        for k in kwargs:
            try:
                kwargs[k].load_state_dict(ckp[k])
            except RuntimeError:
                torch.nn.DataParallel(kwargs[k]).load_state_dict(ckp[k])
        return start_epoch


class Logger(object):
    def __init__(self, quiet=False, log_fn=None, rank=0, prefix=""):
        self.rank = rank if rank is not None else 0
        self.quiet = quiet
        self.log_fn = log_fn

        self.prefix = ""
        if prefix:
            self.prefix = prefix + ' | '

        self.file_pointers = []
        if self.rank == 0:
            if self.quiet:
                open(log_fn, 'w').close()

    def add_line(self, content):
        if self.rank == 0:
            msg = self.prefix+content
            if self.quiet:
                fp = open(self.log_fn, 'a')
                fp.write(msg+'\n')
                fp.flush()
                fp.close()
            else:
                print(msg)
                sys.stdout.flush()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def parameter_description(model):
    desc = ''
    for n, p in model.named_parameters():
        desc += "{:70} | {:10} | {:30} | {}\n".format(
            n, 'Trainable' if p.requires_grad else 'Frozen',
            ' x '.join([str(s) for s in p.size()]), str(np.prod(p.size())))
    return desc


def synchronize_meters(progress, cur_gpu):
    metrics = torch.tensor([m.avg for m in progress.meters]).cuda(cur_gpu)
    metrics_gather = [torch.ones_like(metrics) for _ in range(dist.get_world_size())]
    dist.all_gather(metrics_gather, metrics)

    metrics = torch.stack(metrics_gather).mean(0).cpu().numpy()
    for meter, m in zip(progress.meters, metrics):
        meter.avg = m


def plot_loss(path, history):
    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['err'],
             color='b', label='training')
    plt.legend()
    fig.savefig(os.path.join(path, 'loss.png'), dpi=200)
    plt.close('all')


def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid

def recover_rgb(img):
    for t, m, s in zip(img,
                       [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    return img


def magnitude2heatmap(mag, log=True, scale=200.):
    if log:
        mag = np.log10(mag + 1.)
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    mag_color = mag_color[:, :, ::-1]
    return mag_color


def istft_reconstruction(mag, phase, hop_length=256):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)


class VideoWriter:
    """ Combine numpy frames into video using ffmpeg
    Arguments:
        filename: name of the output video
        fps: frame per second
        shape: shape of video frame
    Properties:
        add_frame(frame):
            add a frame to the video
        add_frames(frames):
            add multiple frames to the video
        release():
            release writing pipe
    """

    def __init__(self, filename, fps, shape):
        self.file = filename
        self.fps = fps
        self.shape = shape

        # video codec
        ext = filename.split('.')[-1]
        if ext == "mp4":
            self.vcodec = "h264"
        else:
            raise RuntimeError("Video codec not supoorted.")

        # video writing pipe
        cmd = [
            "ffmpeg",
            "-y",                                     # overwrite existing file
            "-f", "rawvideo",                         # file format
            "-s", "{}x{}".format(shape[1], shape[0]), # size of one frame
            "-pix_fmt", "rgb24",                      # 3 channels
            "-r", str(self.fps),                      # frames per second
            "-i", "-",                                # input comes from a pipe
            "-an",                                    # not to expect any audio
            "-vcodec", self.vcodec,                   # video codec
            "-pix_fmt", "yuv420p",                  # output video in yuv420p
            self.file]

        self.pipe = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE, bufsize=10**9)

    def release(self):
        self.pipe.stdin.close()

    def add_frame(self, frame):
        assert len(frame.shape) == 3
        assert frame.shape[0] == self.shape[0]
        assert frame.shape[1] == self.shape[1]
        try:
            self.pipe.stdin.write(frame.tostring())
        except:
            _, ffmpeg_error = self.pipe.communicate()
            print(ffmpeg_error)

    def add_frames(self, frames):
        for frame in frames:
            self.add_frame(frame)


def kill_proc(proc):
    proc.kill()
    print('Process running overtime! Killed.')


def run_proc_timeout(proc, timeout_sec):
    # kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        proc.communicate()
    finally:
        timer.cancel()


def combine_video_audio(src_video, src_audio, dst_video, verbose=False):
    try:
        cmd = ["ffmpeg", "-y",
               "-loglevel", "quiet",
               "-i", src_video,
               "-i", src_audio,
               "-c:v", "copy",
               "-c:a", "aac",
               "-strict", "experimental",
               dst_video]
        proc = sp.Popen(cmd)
        run_proc_timeout(proc, 10.)

        if verbose:
            print('Processed:{}'.format(dst_video))
    except Exception as e:
        print('Error:[{}] {}'.format(dst_video, e))


# save video to the disk using ffmpeg
def save_video(path, tensor, fps=25):
    assert tensor.ndim == 4, 'video should be in 4D numpy array'
    L, H, W, C = tensor.shape
    writer = VideoWriter(
        path,
        fps=fps,
        shape=[H, W])
    for t in range(L):
        writer.add_frame(tensor[t])
    writer.release()


def save_audio(path, audio_numpy, sr):
    librosa.output.write_wav(path, audio_numpy, sr)


def IOU(boxA, boxB):
    '''
        round(bbox.left * width),
        round(bbox.top * height),
        round(bbox.right * width),
        round(bbox.bottom * height),
    '''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def find_bbox_in_heatmap(prob_map, size, thresh=0.5):
    """
    Args:
        prob_map (np.ndarry) (HxW)
        size = (tuple)
    """
    prob_map[prob_map < thresh] = 0
    prob_map = np.uint8(prob_map * 255)

    xtl, ytl, xbr, ybr = math.inf, math.inf, -math.inf, -math.inf
    thresh = cv2.threshold(prob_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= 100]
    bbox = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # bbox.append(np.array([x, y, x+w, y+h], dtype=np.int))
        xtl, ytl = min(xtl, x), min(ytl, y)
        xbr, ybr = max(xbr, x + w), max(ybr, y + h)
    if xtl == math.inf:
        bbox.append(np.array([0, 0, 456, 256], dtype=np.int))
    else:
        bbox.append(np.array([xtl, ytl, xbr, ybr], dtype=np.int))

    return bbox


def save_img_with_bbox(img, bbox, path):
    '''
    Args:
        img: cv2 image
        bbox: [int]
    '''
    img = np.array(img)
    img = img[:, :, ::-1].copy() 
    for i in range(len(bbox)):
        bb.add(img, bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3], color="red")
    cv2.imwrite(path, img)


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    return value


def apply_heatmap(img, prob_map, size, bbox=False, thresh=0.5):
    """
    Args:
        image (PIL.Image)
        prob_map (numpy.ndarray)
        size (length2 tuple)
    """
    prob_map[prob_map < thresh] = 0

    gray = np.uint8(prob_map * 255)
    prob_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    prob_map = cv2.cvtColor(prob_map, cv2.COLOR_BGR2RGB)
    prob_map = np.float32(prob_map) / 255

    cam = prob_map + np.float32(img) / 255
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cam = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)

    return cam


def magnitude2heatmap(mag, log=True, scale=200.):
    if log:
        mag = np.log10(mag + 1.)
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    mag_color = mag_color[:, :, ::-1]
    return mag_color


def istft_reconstruction(mag, phase, hop_length=256):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)


def generate_gaussian_mask(raw_mask):
    '''
        raw_mask: [B, 1, H, W]

    '''
    B, _, H, W = raw_mask.shape
    raw_mask = raw_mask.squeeze(1)
    num_pixels = H*W
    gaussian_mask = []
    for idx in range(B):
        # print(idx)
        index = (raw_mask[idx] != 0).nonzero(as_tuple=True)
        if index[0].shape[0] == 0:
            g = torch.ones(H,W) 
            gaussian_mask.append(g)
            continue
        xl = index[0][0]
        yl = index[1][0]
        xr = index[0][-1]
        yr = index[1][-1]
        x0 = int((xl + xr) / 2)
        y0 = int((yl + yr) / 2)

        sigma = 1

        x, y = torch.arange(H), torch.arange(W)

        gx = torch.exp(-(x-x0)**2/(2*sigma**2))
        gy = torch.exp(-(y-y0)**2/(2*sigma**2))
        g = torch.outer(gx, gy)
        g /= torch.sum(g)  # normalize, if you want that

        gaussian_mask.append(g)

    gaussian_mask = torch.stack(gaussian_mask).cuda()
    gaussian_mask = gaussian_mask.unsqueeze(1)

    return gaussian_mask


def generate_gaussian_mask_centers(center_x, center_y, H, W, sigma=2):
    '''
        center_x: [B, 1]

    '''
    B = center_x.shape[0]
    gaussian_mask = []
    for idx in range(B):
        x0 = center_x[idx]
        y0 = center_y[idx]
        x, y = torch.arange(H).cuda(), torch.arange(W).cuda()
        gx = torch.exp(-(x-x0)**2/(2*sigma**2))
        gy = torch.exp(-(y-y0)**2/(2*sigma**2))
        g = torch.outer(gx, gy)
        gaussian_mask.append(g)

    gaussian_mask = torch.stack(gaussian_mask).cuda()
    gaussian_mask = gaussian_mask.unsqueeze(1)

    return gaussian_mask


def homo(img1, img2):
    '''
    predict the homography from img1->img2 and warp img1 to img2's perspective
    '''
    MIN_MATCH_COUNT = 10
    # img1 = cv2.imread(img1_file)          # queryImage
    # img2 = cv2.imread(img2_file)          # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    if (len(kp1) < 2) or (len(kp2) < 2):
        return torch.eye(3)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    if len(good) < 1:
        return torch.eye(3)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        if M is None:
            return torch.eye(3)
        return torch.from_numpy(M)
    else:
        return torch.eye(3)


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor2cv(tensor):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return image


def noramlize(batch):
    B, C, H, W = batch.size()
    batch = batch.view(batch.size(0), -1)
    batch -= batch.min(1, keepdim=True)[0]
    batch /= batch.max(1, keepdim=True)[0]
    batch = batch.view(B, C, H, W)
    return batch


def np_normalize(img):
    '''
    normailze to [0,1]
    '''
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def cv2to255(img):
    norm_image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    norm_image = norm_image.astype(np.uint8)


def eval_cal_ciou_all(heat_map, gt_map, img_size=224, THRES=None):

    # preprocess heatmap
    heat_map = cv2.resize(heat_map, dsize=(456, 256), interpolation=cv2.INTER_LINEAR)
    heat_map = normalize_img(heat_map)
    # print(heat_map)
    CIOU = []

    for thres in THRES:
        # convert heatmap to mask
        pred_map = heat_map
        if thres is None:
            threshold = np.sort(pred_map.flatten())[int(pred_map.shape[0] * pred_map.shape[1] / 2)]
            pred_map[pred_map >= threshold] = 1
            pred_map[pred_map < 1] = 0
            infer_map = pred_map
        else:
            infer_map = np.zeros((256, 456))
            infer_map[pred_map >= thres] = 1

        # compute ciou
        inter = np.sum(infer_map * gt_map)
        union = np.sum(gt_map) + np.sum(infer_map * (gt_map == 0))
        ciou = inter / union
        # print(ciou)
        CIOU.append(ciou)
    return CIOU, inter, union


def eval_ap(heat_map, gt_map):
    # preprocess heatmap
    heat_map = cv2.resize(heat_map, dsize=(256, 456), interpolation=cv2.INTER_LINEAR)
    heat_map = normalize_img(heat_map)

    heat_map = heat_map.flatten()
    gt_map = gt_map.flatten()

    ap = average_precision_score(gt_map, heat_map)

    return ap


def testset_gt_epic(label, label_id, save=False, img=None, path=None):
    file = open(label_id, 'r')
    lines = file.readlines()
    if label - 1 >= len(lines):
        return None
    line = lines[label-1]
    item_ = line[:-1].split(' ')[1:]
    gt_map = np.zeros([256, 456])

    (xmin, ymin, xmax, ymax) = int(item_[0]), int(item_[1]), int(item_[2]), int(item_[3])
    if save:
        img = np.array(img)
        img = np.uint8(img)
        img = img[:, :, ::-1].copy() 
        bb.add(img, xmin, ymin, xmax, ymax, color="red")
        cv2.imwrite(path, img)
        preannotated_img_name = label_id[:-3] + 'jpg'
        pre_path = path.replace('step','step_pre')
        cmd = "cp %s %s"%(preannotated_img_name, pre_path)
        os.system(cmd)

    temp = np.zeros([256, 456])
    temp[ymin:ymax, xmin:xmax] = 1
    gt_map += temp
    gt_map[gt_map > 0] = 1
    return gt_map
