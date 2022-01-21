#!/bin/bash

cd /root/kaggle/yolov5
python3 val.py --img 4000 --data /root/gbr/yolov5/reef_f1_naive.yaml --weights /root/kaggle/yolov5/runs/train/l6_3600_uflip_vm5_f2/weights/best.pt --conf-thres 0.01 --iou-thres 0.3 --save-txt --save-conf --exist-ok