import random
import os
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import torchaudio
import librosa, librosa.display
import soundfile as sf
from pathlib import Path
from PIL import Image
from scipy import signal
import pandas as pd
import sys
from datasets import video_transforms as vtransforms
from pydub import AudioSegment
import soundfile as sf
from glob import glob

class BaseDataset(torchdata.Dataset):
    def __init__(self, list_sample, max_sample=-1, split='train'):
        # params
        self.num_frames = 3
        self.stride_frames = 10
        self.frameRate = 50
        self.imgSize = 224
        self.audRate = 11025
        self.audLen = 11025
        self.audSec = 1. * self.audLen / self.audRate

        # STFT params
        self.log_freq = 1
        self.stft_frame = 1022
        self.stft_hop = 256
        self.HS = self.stft_frame // 2 + 1
        self.WS = (self.audLen + 1) // self.stft_hop

        self.split = split
        self.seed = 1234
        random.seed(self.seed)

        # initialize video transform
        self._init_vtransform() 
        self._init_atransform()  

        # list_sample can be a python list or a csv file of list
        if isinstance(list_sample, str):
            self.list_sample = []
            for row in csv.reader(open(list_sample, 'r'), delimiter=','):
                # if len(row) < 2:
                    # continue
                self.list_sample.append(row)
    
        elif isinstance(list_sample, list):
            self.list_sample = list_sample
        else:
            raise('Error list_sample!')

        self.list_sample = self.list_sample[1:]

        if self.split == 'train':
            random.shuffle(self.list_sample)
            self.list_sample = self.list_sample[0:max_sample]

        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def __len__(self):
        return len(self.list_sample)

    def string2list(self, str, id=False):
        if not id:
            return str.split(',')
        else:
            return str.replace(' ', '')[1:-1].split(',')

    def resize_bbox(self, bbox, height: int, width: int):
        return (
            round(bbox.left * width),
            round(bbox.top * height),
            round(bbox.right * width),
            round(bbox.bottom * height),
        )

    # video transform funcs
    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            transform_list.append(vtransforms.Resize((int(self.imgSize * 1.1), int(self.imgSize * 1.1)), transforms.InterpolationMode.BICUBIC))
            transform_list.append(vtransforms.ToTensor())
            transform_list.append(vtransforms.Normalize(mean, std))
            transform_list.append(vtransforms.Stack())
            transform_list.append(vtransforms.RandomCropVideo(self.imgSize))
        else:
            transform_list.append(vtransforms.Resize((self.imgSize, self.imgSize), transforms.InterpolationMode.BICUBIC))
            transform_list.append(vtransforms.CenterCrop(self.imgSize))
            transform_list.append(vtransforms.ToTensor())
            transform_list.append(vtransforms.Normalize(mean, std))
            transform_list.append(vtransforms.Stack())


        self.vid_transform = transforms.Compose(transform_list)
        self.mask_transform = transforms.Compose([vtransforms.Resize(self.imgSize, transforms.InterpolationMode.BICUBIC),\
        vtransforms.CenterCrop(self.imgSize), vtransforms.ToTensor(), vtransforms.Stack()])

    # image transform funcs, deprecated
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            self.img_transform = transforms.Compose([
                transforms.Scale(int(self.imgSize)),
                #transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Scale(self.imgSize),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
    
    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])

    def _load_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame(path))
        frames = self.vid_transform(frames)
        return frames

    # load frame using PIL
    def _load_frame(self, path):
        if not os.path.exists(path):
            img = Image.new("RGB", size=(456, 256))
        else:
            img = Image.open(path).convert('RGB')
        if img.size != (456, 256):
            img = img.resize((456, 256))
        return img

    # stft
    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio_file(self, path):
        if path.endswith('.mp3'):
            audio_raw, rate = torchaudio.load(path)
            audio_raw = audio_raw.numpy().astype(np.float32)

            # range to [-1, 1]
            audio_raw *= (2.0**-31)

            # convert to mono
            if audio_raw.shape[1] == 2:
                audio_raw = (audio_raw[:, 0] + audio_raw[:, 1]) / 2
            else:
                audio_raw = audio_raw[:, 0]
        else:
            audio_raw, rate = librosa.load(path, sr=None, mono=True)
            '''
            audio_raw, rate = torchaudio.load(path)
            if audio_raw.size(0)>1:
                audio_raw = torch.mean(audio_raw, dim=0)
            audio_raw = audio_raw.view(-1)
            audio_raw = audio_raw.numpy().astype(np.float32)
            '''

        return audio_raw, rate

    def _load_audio(self, folder, nearest_resample=False):
        audio = np.zeros(self.audLen, dtype=np.float32)

        path_list = list(Path(folder).glob('audio/*.wav'))
        path = str(random.choice(path_list))
        # path = glob(os.path.join(folder, '*.wav'))[0]

        # load audio
        audio_raw, rate = self._load_audio_file(path)
        # audio_raw = np.zeros(*audio_raw.shape, dtype=np.float32)
        # audio_raw, rate = self.audios[v_id]

        # repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audSec:
            n = int(rate * self.audSec / (audio_raw.shape[0] + 1e-10)) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.audRate:
            # print('resmaple {}->{}'.format(rate, self.audRate))
            if nearest_resample:
                audio_raw = audio_raw[::rate//self.audRate]
            else:
                audio_raw = librosa.resample(audio_raw, rate, self.audRate)

        # crop N seconds
        len_raw = audio_raw.shape[0]
        # center = int(center_timestamp * self.audRate)
        center = len_raw // 2
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)

        audio[self.audLen//2-(center-start): self.audLen//2+(end-center)] = \
            audio_raw[start:end]

        # randomize volume
        if self.split == 'train':
            scale = random.random() + 0.5     # 0.5-1.5
            audio *= scale
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.

        return audio, rate
    
    def augment_audio(self, audio):
        audio = audio * (random.random() + 0.5) # 0.5 - 1.5
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.
        return audio
    
    def generate_spectrogram_magphase(self, audio, stft_frame=254, stft_hop=64, with_phase=True):
        spectro = librosa.core.stft(audio, n_fft=stft_frame, hop_length=stft_hop, center=True)
        spectro_mag, spectro_phase = librosa.core.magphase(spectro)
        spectro_mag = np.expand_dims(spectro_mag, axis=0)
        if with_phase:
            spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
            return spectro_mag, spectro_phase
        else:
            return spectro_mag



class EpicDataset(BaseDataset):
    def __init__(self, list_sample, args, **kwargs):
        super(EpicDataset, self).__init__(
            list_sample, **kwargs)
        self.fps = 50
        self.audLen = 11025
        self.num_frames = args.num_frames

    def __getitem__(self, index):
        img_list = []
        audio_mixs = [None for n in range(2)] #audios of mixed videos

        infoN = self.list_sample[index]

        if self.split == 'train':
            p_id, v_id, start_time, end_time, start_frame, end_frame, narration, folder_dir = infoN
            indexO = random.randint(0, len(self.list_sample) - 1)
            while indexO == index:
                indexO = random.randint(0, len(self.list_sample) - 1)
            _, _, _, _, _, _, _, folder_dir_Other = self.list_sample[indexO]
        else:
            folder_dir, frame_ids = infoN[:2]
            frame_ids = self.string2list(frame_ids, True)
            v_id, start_time, end_time, start_frame, end_frame, _ = folder_dir.rsplit('/', 1)[1].split('-')
            p_id = v_id.split('_')[0]

            # sample the second audio file
            indexO = random.randint(0, len(self.list_sample) - 1)
            folder_dir_Other, _, _, _, _, \
                _, _, _ = self.list_sample[indexO]

        start_frame = int(float(start_frame))
        end_frame = int(float(end_frame))
        count_framesN = end_frame - start_frame

        center_frameN = int(count_framesN) // 2 + 1 + start_frame
        
        interval = 2

        if self.split == 'train':
            frame_ids = []
            delta = int(self.num_frames//2)
            query_frame = center_frameN

            for i in range(self.num_frames+1):
                frame_to_add = (i-delta)*interval+query_frame
                if frame_to_add < start_frame:
                    frame_ids.append(start_frame)
                elif frame_to_add > end_frame:
                    frame_ids.append(end_frame-1)
                else:
                    frame_ids.append(frame_to_add)

            for i in range(len(frame_ids)):
                img = self._load_frame(os.path.join("YOUR_DATA_DIR/%s/rgb_frames"%p_id,
                            v_id,
                            'frame_{:010d}.jpg'.format(frame_ids[i])))
                img_list.append(img)
            img_transform = self.vid_transform(img_list)  
            img_list = [np.array(i) for i in img_list] 
        else: 

            self.num_frames = 8
            frame_ids = []
            delta = int(self.num_frames//2)
            query_frame = center_frameN

            for i in range(self.num_frames+1):
                frame_to_add = (i-delta)*interval+query_frame
                if frame_to_add < start_frame:
                    frame_ids.append(start_frame)
                elif frame_to_add > end_frame:
                    frame_ids.append(end_frame-1)
                else:
                    frame_ids.append(frame_to_add)

            for i in range(len(frame_ids)):
                img = self._load_frame(os.path.join("YOUR_DATA_DIR/%s/rgb_frames"%p_id,
                            v_id,
                            'frame_{:010d}.jpg'.format(frame_ids[i])))
                img_list.append(img)
            img_transform = self.vid_transform(img_list)  
            img_list = [np.array(i) for i in img_list]   


        # load frames and audios, STFT
        audios, rate = self._load_audio(folder_dir)
        audios_mag, audios_phase = self.generate_spectrogram_magphase(audios)
        audios_other, rate_other = self._load_audio(folder_dir_Other)
        audios_mag_other, audios_phase_other = self.generate_spectrogram_magphase(audios_other)
        audio_mixs[0] = audios_mag
        audio_mixs[1] = audios_mag_other
        audio_mix = (audios + audios_other) / 2
        audios_mix_mag, audios_mix_phase = self.generate_spectrogram_magphase(audio_mix)
       
        spectrogram = torch.FloatTensor(audios_mix_mag)
        spec_all = torch.FloatTensor(np.array(audio_mixs))

        if self.split == 'train':
            return img_transform, spectrogram, img_list, spec_all
        else:
            return img_transform, spec_all[0, :, :, :], img_list, spec_all, folder_dir
