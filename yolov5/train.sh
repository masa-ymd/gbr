#!/bin/bash

cd /root/kaggle/yolov5
python3 train.py --img 2400 --batch 5 --epochs 15 --data /root/gbr/yolov5/reef_90_10_1982.yaml --weights yolov5s6.pt --name split_90_10_5s6_onlyano_ --hyp /root/gbr/yolov5/hyp.heavy.2.yaml
#python3 train.py --img 3000 --batch 2 --epochs 15 --data /root/gbr/yolov5/reef_f1_naive.yaml --weights yolov5m6.pt --name l6_3600_uflip_vm5_m_f2 --hyp /root/gbr/yolov5/hyp.heavy.2.yaml
#python3 train.py --img 4000 --batch 4 --epochs 20 --data /root/gbr/yolov5/reef_f1_naive.yaml --weights /root/kaggle/tensorflow-great-barrier-reef/models/pretrained_weights/best_f1.pt --freeze 10 --name f1_fineturning --hyp /root/gbr/yolov5/hyp.heavy.2.yaml
#python3 train.py --img 3000 --batch 2 --epochs 15 --data /root/gbr/yolov5/reef_80_20_naive.yaml --weights yolov5m.pt --name split_80_20_5m --hyp /root/gbr/yolov5/hyp.heavy.2.yaml
#python3 train.py --img 3000 --batch 2 --epochs 15 --data /root/gbr/yolov5/reef_80_20_naive.yaml --weights /root/kaggle/yolov5/runs/train/split_80_20_5m62/weights/best.pt --name split_80_20_5m --hyp /root/gbr/yolov5/hyp.heavy.2.yaml
#python3 train.py --img 2400 --batch 2 --epochs 2 --data /root/gbr/yolov5/reef_90_10_naive.yaml --weights /root/kaggle/tensorflow-great-barrier-reef/models/pretrained_weights/last_5l6_e6.pt --name split_90_10_5l6_finetune --freeze 10 --hyp /root/gbr/yolov5/hyp.finetune.yaml