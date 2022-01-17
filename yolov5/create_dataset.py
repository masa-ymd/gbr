import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from shutil import copyfile

FOLD = 1
DATA_PATH = '/root/kaggle/tensorflow-great-barrier-reef/data'
YOLO_DATA_PATH = f'{DATA_PATH}/yolo_data/fold{FOLD}/'

train = pd.read_csv(f'{DATA_PATH}/train.csv')
train['pos'] = train.annotations != '[]'