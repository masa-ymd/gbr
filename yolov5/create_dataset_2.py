import ast
import os
from shutil import copyfile
from tqdm import tqdm
tqdm.pandas()
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


BASE_DIR = '/root/kaggle/tensorflow-great-barrier-reef'
DATA_DIR = f'{BASE_DIR}/yolo_spilt9010_dataset'

df = pd.read_csv(f"{BASE_DIR}/data/reef-cv-strategy-subsequences-dataframes/train-validation-split/train-0.1.csv")
df['image_path'] = df['image_path'].str.replace('../input/tensorflow-great-barrier-reef', f'{BASE_DIR}/data')
print(df.head(3))

def add_new_path(row):
    if row.is_train:
        return f"{DATA_DIR}/images/train/{row.image_id}.jpg"
    else:
        return f"{DATA_DIR}/images/valid/{row.image_id}.jpg"
    
df['new_path'] = df.apply(lambda row: add_new_path(row), axis=1)
print("New image path for train/valid created")
print(df.head(3))

os.makedirs(f"{DATA_DIR}/images/train", exist_ok=True)
os.makedirs(f"{DATA_DIR}/images/valid", exist_ok=True)
os.makedirs(f"{DATA_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{DATA_DIR}/labels/valid", exist_ok=True)
print(f"Directory structure for Yolov5 created")
    
_ = df.progress_apply(lambda row: copyfile(row.image_path, row.new_path), axis=1)
print("Sucessfully copy file for train and valid")


IMG_WIDTH, IMG_HEIGHT = 1280, 720
def get_yolo_format_bbox(bbox, img_w, img_h):
    w = bbox['width']
    h = bbox['height']
    
    if (bbox['x'] + bbox['width'] > img_w):
        w = img_w - bbox['x']
    if (bbox['y'] + bbox['height'] > img_h):
        h = img_h - bbox['y']
    
    xc = bbox['x'] + int(np.round(w/2))
    yc = bbox['y'] + int(np.round(h/2))
    
    # normalize
    return [xc/img_w, yc/img_h, w/img_w, h/img_h]

for index, row in tqdm(df.iterrows(), total=len(df)):
    annotations = ast.literal_eval(row.annotations)
    bboxes = []
    for annot in annotations:
        bbox = get_yolo_format_bbox(annot, IMG_WIDTH, IMG_HEIGHT)
        bboxes.append(bbox)
        
    if row.is_train:
        file_name = f"{DATA_DIR}/labels/train/{row.image_id}.txt"
    else:
        file_name = f"{DATA_DIR}/labels/valid/{row.image_id}.txt"
        
    with open(file_name, 'w') as f:
        for i, bbox in enumerate(bboxes):
            label = 0
            bbox = [label] + bbox
            bbox = [str(i) for i in bbox]
            bbox = " ".join(bbox)
            f.write(bbox)
            f.write("\n")

print("Annotations in Yolov5 format for all images created.")


train_data = os.listdir(f"{DATA_DIR}/labels/train")
num_train_file = len(train_data)
print("Number of txt file in train folder: ", num_train_file)

valid_data = os.listdir(f"{DATA_DIR}/labels/valid")
num_valid_file = len(valid_data)
print("Number of txt file in valid foler: ", num_valid_file)