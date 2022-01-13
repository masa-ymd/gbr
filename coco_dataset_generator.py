from ast import literal_eval
import pandas as pd
from tqdm import tqdm

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

def coco(df):
    
    annotion_id = 0
    images = []
    annotations = []

    categories = [{'id': 0, 'name': 'cots'}]

    for i, row in tqdm(df.iterrows(), total = len(df)):

        images.append({
            "id": i,
            "file_name": f"{row['image_id']}.jpg",
            "height": 720,
            "width": 1280,
        })
        for bbox in row['annotations']:
            annotations.append({
                "id": annotion_id,
                "image_id": i,
                "category_id": 0,
                "bbox": list(bbox.values()),
                "area": bbox['width'] * bbox['height'],
                "segmentation": [],
                "iscrowd": 0
            })
            annotion_id += 1

    json_file = {'categories':categories, 'images':images, 'annotations':annotations}
    return json_file

json_train = coco(df[~df['is_valid']])
json_valid = coco(df[ df['is_valid']])