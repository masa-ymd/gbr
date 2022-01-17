#!/bin/bash

python /root/kaggle/yolov5/train.py --img 3000 --batch 4 --epochs 5 --data ./reef_f1_naive.yaml --weights yolov5s6.pt --name l6_3600_uflip_vm5_f1 --hyp ./hyp.heavy.2.yaml