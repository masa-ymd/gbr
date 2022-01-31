#!/bin/bash

cd /root/kaggle/yolov5
python3 val.py --imgsz 6000 --batch 2 --weights /root/kaggle/yolov5/runs/train/split_80_20_5m63/weights/best.pt --data /root/gbr/yolov5/reef_80_20_naive.yaml --augment --conf-thres 0.28 --iou-thres 0.4 --save-txt --save-conf --exist-ok