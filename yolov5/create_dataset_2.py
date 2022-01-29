import os
from shutil import copyfile
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

BASE_DIR = '/root/kaggle/tensorflow-great-barrier-reef'
DATA_DIR = f'{BASE_DIR}/yolo_spilt_dataset'

df = pd.read_csv(f"{BASE_DIR}/data/reef-cv-strategy-subsequences-dataframes/train-validation-split/train-0.2.csv")
df['image_path'] = df['image_path'].str.replace('../input/tensorflow-great-barrier-reef', f'{BASE_DIR}')
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
print(f"Directory structure yor Yolov5 created")

#def copy_file(row):
#    copyfile(row.image_path, row.new_path)
    
_ = df.progress_apply(lambda row: copyfile(row.image_path, row.new_path), axis=1)
print("Sucessfully copy file for train and valid")