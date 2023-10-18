#!/bin/bash
OPTS=""
OPTS+="--id MODEL_ID "
OPTS+="--list_train YOUR_TRAIN_FILE "
OPTS+="--list_val YOUR_TEST_FILE "

# Models
OPTS+="--arch_sound vggish "
OPTS+="--arch_frame drn "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 32 "

# weights
# OPTS+="--weights_sound '' "
# OPTS+="--weights_frame '' "
# OPTS+="--weights_tnet '' "

# binary mask, BCE loss, weighted loss
OPTS+="--loss l2 "
OPTS+="--weighted_loss 0 "

# logscale in frequency
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 4 "
OPTS+="--stride_frames 2 "
OPTS+="--frameRate 50 "

# audio-related
OPTS+="--audLen 11025 " # 11025
OPTS+="--audRate 11025 " # 11025

# learning params
OPTS+="--num_gpus 1 "
OPTS+="--workers 4 "
OPTS+="--batch_size 8 "
OPTS+="--batch_size_per_gpu 16 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--num_epoch 40 "
OPTS+="--lr_steps 20 30 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_val 256 "
OPTS+="--num_vis 40 "
OPTS+="--mode train "

python -u ../main_localization.py $OPTS