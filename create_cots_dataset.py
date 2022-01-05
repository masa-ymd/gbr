import warnings
warnings.filterwarnings("ignore")
import pandas as pd

TRAIN_PATH = '/root/kaggle/tensorflow-great-barrier-reef/data'


def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

def get_path(row):
    row['image_path'] = f'{TRAIN_PATH}/train_images/video_{row.video_id}/{row.video_frame}.jpg'
    return row

df = pd.read_csv(f"{TRAIN_PATH}/train.csv")
df.head(5)