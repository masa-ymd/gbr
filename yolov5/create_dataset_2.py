import pandas as pd
pd.set_option("display.max_columns", None)

BASE_DIR = '/root/kaggle/tensorflow-great-barrier-reef'

df = pd.read_csv(f"{BASE_DIR}/data/reef-cv-strategy-subsequences-dataframes/train-validation-split/train-0.2.csv")
df['image_path'] = df['image_path'].str.replace('../input/tensorflow-great-barrier-reef', f'{BASE_DIR}')
print(df.head(500))