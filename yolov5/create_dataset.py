import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from shutil import copyfile
from tqdm import tqdm
import os

FOLD = 0
DATA_PATH = '/root/kaggle/tensorflow-great-barrier-reef/data'
YOLO_DATA_PATH = f'{DATA_PATH}/yolo_data/fold{FOLD}/'

if not os.path.exists(f'{YOLO_DATA_PATH}/images/train'):
    os.makedirs(f'{YOLO_DATA_PATH}/images/train')

if not os.path.exists(f'{YOLO_DATA_PATH}/images/val'):
    os.makedirs(f'{YOLO_DATA_PATH}/images/val')

if not os.path.exists(f'{YOLO_DATA_PATH}/labels/train'):
    os.makedirs(f'{YOLO_DATA_PATH}/labels/train')

if not os.path.exists(f'{YOLO_DATA_PATH}/labels/val'):
    os.makedirs(f'{YOLO_DATA_PATH}/labels/val')

train = pd.read_csv(f'{DATA_PATH}/train.csv')
train['pos'] = train.annotations != '[]'

annos = []
for i, x in tqdm(train.iterrows(), total=len(train)):
    if x.video_id == FOLD:
        mode = 'val'
    else:
        # train
        mode = 'train'
        if not x.pos: continue
        # val
    copyfile(f'{DATA_PATH}/train_images/video_{x.video_id}/{x.video_frame}.jpg',
                f'{YOLO_DATA_PATH}/images/{mode}/{x.image_id}.jpg')
    if not x.pos:
        continue
    r = ''
    anno = eval(x.annotations)
    for an in anno:
#            annos.append(an)
        r += '0 {} {} {} {}\n'.format((an['x'] + an['width'] / 2) / 1280,
                                        (an['y'] + an['height'] / 2) / 720,
                                        an['width'] / 1280, an['height'] / 720)
    with open(f'{YOLO_DATA_PATH}/labels/{mode}/{x.image_id}.txt', 'w') as fp:
        fp.write(r)