#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import warnings
warnings.filterwarnings("ignore")

from yolox.exp import Exp as MyExp

TRAIN_PATH = '/root/kaggle/tensorflow-great-barrier-reef/data'
COCO_DATASET_PATH = f'{TRAIN_PATH}/cocodataset'

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # Define yourself dataset path
        self.data_dir = f"{COCO_DATASET_PATH}"
        self.train_ann = "train.json"
        self.val_ann = "valid.json"

        self.num_classes = 1

        self.max_epoch = 20
        self.data_num_workers = 2
        self.eval_interval = 1
        
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.no_aug_epochs = 2
        
        self.input_size = (960, 960)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (960, 960)