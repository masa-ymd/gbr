#!/bin/bash

cd /root/kaggle/yolov5
python train.py --img 3600 --batch 4 --epochs 15 --data /root/gbr/yolov5/reef_f1_naive.yaml --weights yolov5s6.pt --name l6_3600_uflip_vm5_f1 --hyp /root/gbr/yolov5/hyp.heavy.2.yaml