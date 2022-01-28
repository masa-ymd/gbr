#!/bin/bash

cd /root/kaggle/yolov5
python3 val.py --imgsz 3600 --batch 2 --data /root/gbr/yolov5/reef_f1_naive.yaml --weights /root/kaggle/yolov5/runs//train/f1_fineturning8/weights/best.pt --conf-thres 0.01 --iou-thres 0.3 --save-txt --save-conf --exist-ok