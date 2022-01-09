#!/bin/bash

python3 -B train.py \
-f /root/gbr/yolox/cots_config_yolox_x.py \
-d 1 \
-b 32 \
--fp16 \
-o \
-c /root/kaggle/tensorflow-great-barrier-reef/models/pretrained_weights/yolox_x.pth