from ast import literal_eval
import pandas as pd
from tqdm import tqdm

TRAIN_PATH = '/root/kaggle/tensorflow-great-barrier-reef/data/train_images'
#N_SAMP = 6000
#SEED = 42

df = pd.read_csv('/root/kaggle/tensorflow-great-barrier-reef/data/train.csv')
n_with_annotations = (df['annotations'] != '[]').sum()

print(f'n_with_annotations: {n_with_annotations}')

#df = pd.concat([
#    df[df['annotations'] != '[]'],
#    df[df['annotations'] == '[]'].sample(N_SAMP - n_with_annotations, random_state=SEED)
#]).sample(frac=1).reset_index(drop = True)
df = df[df['annotations'] != '[]']

df['is_valid'] = df['video_id'] == 2
df['annotations'] = df['annotations'].apply(literal_eval)
df['path'] = df.apply(lambda row: f"{TRAIN_PATH}/video_{row['video_id']}/{row['video_frame']}.jpg", axis = 1)

with pd.option_context("display.max_columns", 100):
    print(df.tail())

print(f"train: {len(df[~df['is_valid']])}")
print(f"val: {len(df[df['is_valid']])}")

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
            b_width = bbox['width']
            b_height = bbox['height']
            
            # some boxes in COTS are outside the image height and width
            if (bbox['x'] + bbox['width'] > 1280):
                b_width = 1280 - bbox['x'] 
            if (bbox['y'] + bbox['height'] > 720):
                b_height = 720 - bbox['y']
            
            annotations.append({
                "id": annotion_id,
                "image_id": i,
                "category_id": 0,
                "bbox": [bbox['x'], bbox['y'], b_width, b_height],
                "area": bbox['width'] * bbox['height'],
                "segmentation": [],
                "iscrowd": 0
            })
            annotion_id += 1

    json_file = {'categories':categories, 'images':images, 'annotations':annotations}
    return json_file

json_train = coco(df[~df['is_valid']])
json_valid = coco(df[ df['is_valid']])

import json

OUTPUT_PATH = '/root/kaggle/tensorflow-great-barrier-reef/data/cocodataset'

with open(f'{OUTPUT_PATH}/annotations/annotations_train.json', 'w', encoding='utf-8') as f:
    json.dump(json_train, f, ensure_ascii=True, indent=4)
    
with open(f'{OUTPUT_PATH}/annotations/annotations_valid.json', 'w', encoding='utf-8') as f:
    json.dump(json_valid, f, ensure_ascii=True, indent=4)

import os
os.makedirs(f'{OUTPUT_PATH}/train2017', exist_ok=True)
os.makedirs(f'{OUTPUT_PATH}/val2017', exist_ok=True)

import shutil
for i, row in tqdm(df.iterrows(), total = len(df)):
    base_dir = 'val2017' if row['is_valid'] else 'train2017'
    fname = f"{row['image_id']}.jpg"
    shutil.copyfile(row['path'], f'{OUTPUT_PATH}/{base_dir}/{fname}')
