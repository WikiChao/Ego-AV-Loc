# Egocentric Audio-Visual Object Localization 
This is the PyTorch implementation of the paper "Egocentric Audio-Visual Object Localization."
 
  
## Introduction
We propose the first joint audio-video generation framework named MM-Diffusion that brings engaging watching and listening experiences simultaneously, towards high-quality realistic videos.  MM-Diffusion consists of a sequential multi-modal U-Net. Two subnets for audio and video learn to gradually generate aligned audio-video pairs from Gaussian noises.

<img src="./fig/teaser.png" width=100%>


### Overview
<img src="./fig/MM-UNet2.png" width=100%>


### Visualize
The generated audio-video examples on landscape:

https://user-images.githubusercontent.com/105475691/207589456-52914a01-1175-4f77-b8f5-112d97013f7c.mp4

The generated audio-video examples on AIST++:

https://user-images.githubusercontent.com/105475691/207589611-fe300424-e5e6-4379-a917-d9a07e9dd8fb.mp4

The generated audio-video examples on Audioset:

https://user-images.githubusercontent.com/105475691/207589639-0a371435-f207-4ff4-a78e-3e9c0868d523.mp4

## Citation
If you find our work useful for your research, please consider citing our paper. :blush:
```
@article{ruan2022mmdiffusion,
author = {Ruan, Ludan and Ma, Yiyang and Yang, Huan and He, Huiguo and Liu, Bei and Fu, Jianlong and Yuan, Nicholas Jing and Jin, Qin and Guo, Baining},
title = {MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation},
journal={arXiv preprint},
year = {2022},
month = {December}
}
```

## Contact
If you meet any problems, please describe them in issues or contact:
* Ludan Ruan: <ruanld@ruc.edu.cn> 
* Huan Yang: <huayan@microsoft.com>
