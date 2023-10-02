# Egocentric Audio-Visual Object Localization 
This is the PyTorch implementation of the paper "<a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Egocentric_Audio-Visual_Object_Localization_CVPR_2023_paper.pdf">Egocentric Audio-Visual Object Localization</a>."
 
## Overview
<p align="center">
 <img align="center" src="./fig/motivation.png" width=70%>
</p>

We explore the task of egocentric audio-visual object localization, which aims to localize objects that emit sounds in the first-person recordings. In this work, we propose a new framework to address the uniqueness of egocentric videos by answering the following two questions: (1) how to associate visual content with audio representations while out-of-view sounds may exist; (2) how to persistently associate audio features with visual content that are captured under different viewpoints.

## Epic Sounding Object dataset
Note, some videos are further filtered out and some bounding boxes are updated recently.

### Prepare Dataset
1. Download videos.

    a. Download Epic-Kitchens dataset from: https://epic-kitchens.github.io/2023 (The website provides scirt to download videos)

2. Preprocess videos. 

    a. Trim the video using Epic-Kitchens' original annotations, for example, the test video timestamps can be found at https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/master/EPIC_100_test_timestamps.csv.

    
    b. Extract waveforms at 11000Hz for all the videos.

3. Data splits. Please follow the same train/test splits at https://github.com/epic-kitchens/epic-kitchens-100-annotations

### Annotation Format
The annotations can be found at ```./data/soundingobject.json```.

  * `video` contains the index to locate the segment from a long video. For example, `P04_105-00:05:26.32-00:05:28.01-16316-16400` represents the `video_id,narration_timestamp,start_timestamp,stop_timestamp,start_frame,stop_frame` in the test split csv file.
* `frame` is the exact frame index we use to annotate the sounding object.
* `bbox` is the relative coordinates of bounding box, which is in `[left, top, right, bottom]` format.

## Citation
If you find our work useful for your research, please consider citing our paper. :smile:
```
@inproceedings{huang2023egocentric,
  title={Egocentric Audio-Visual Object Localization},
  author={Huang, Chao and Tian, Yapeng and Kumar, Anurag and Xu, Chenliang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22910--22921},
  year={2023}
}
```
