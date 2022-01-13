from ast import literal_eval
import pandas as pd
from tqdm.notebook import tqdm

TRAIN_PATH = '/root/kaggle/tensorflow-great-barrier-reef/data/train_images'
N_SAMP = 6000

df = pd.read_csv('/root/kaggle/tensorflow-great-barrier-reef/data/train.csv')
n_with_annotations = (df['annotations'] != '[]').sum()

df = pd.concat([
    df[df['annotations'] != '[]'],
    df[df['annotations'] == '[]'].sample(N_SAMP - n_with_annotations)
]).sample(frac=1).reset_index(drop = True)

df['is_valid'] = df['video_id'] == 2
df['annotations'] = df['annotations'].apply(literal_eval)
df['path'] = df.apply(lambda row: f"{TRAIN_PATH}/video_{row['video_id']}/{row['video_frame']}.jpg", axis = 1)

with pd.option_context("display.max_columns", 100):
    print(df.tail())