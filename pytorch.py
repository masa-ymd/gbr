import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection

from tqdm import tqdm  #コマンドラインで実行するとき
# from tqdm.notebook import tqdm  # jupyter で実行するとき

DATA_DIR = "/root/kaggle/competition-name"
MODEL_DIR = "/root/kaggle/competition-name/torch-models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# リソースの指定（CPU/GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 乱数シード固定（再現性の担保）
def fix_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
fix_seed(seed)

# データローダーのサブプロセスの乱数seedが固定
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# データセットの作成
class Mydataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]
        # 前処理などを書く -----

        # --------------------
        return feature, label

# 訓練/テストデータ読み込み
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")


# モデルの定義
class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 64, 3, 2, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())

        self.fc1 = nn.Linear(2 * 2 * 64, 100)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# モデル・損失関数・最適化アルゴリスムの設定
model = Mymodel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

fold = 5

# モデル訓練関数
def train_model(model, train_loader, test_loader):

    X = train.loc[:, features]
    y = train.loc[:, target]

    kf = KFold(n_splits=fold, shuffle=true, random_state=seed)

    for f, (train_idx, valid_idx) in enumerate(kf.split(X=data)):
        train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]
        valid_X, valid_y = X.iloc[valid_idx], y.iloc[valid_idx]

        train_dataset = Mydataset(train_X, train_y)
        valid_dataset = Mydataset(valid_X, valid_y)

        # データローダーの作成
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=16,  # バッチサイズ
                                                shuffle=True,  # データシャッフル
                                                num_workers=2,  # 高速化
                                                pin_memory=True,  # 高速化
                                                worker_init_fn=worker_init_fn
                                                )

        # Train loop ----------------------------
        model.train()  # 学習モードをオン
        train_batch_loss = []
        for data, label in train_loader:
            # GPUへの転送
            data, label = data.to(device), label.to(device)
            # 1. 勾配リセット
            optimizer.zero_grad()
            # 2. 推論
            output = model(data)
            # 3. 誤差計算
            loss = criterion(output, label)
            # 4. 誤差逆伝播
            loss.backward()
            # 5. パラメータ更新
            optimizer.step()
            # train_lossの取得
            train_batch_loss.append(loss.item())

        # Test(val) loop ----------------------------
        model.eval()  # 学習モードをオフ
        test_batch_loss = []
        with torch.no_grad():  # 勾配を計算なし
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = criterion(output, label).item()
                test_batch_loss.append(loss.item())

        # 学習済みモデルの保存
        path_saved_model = f"{MODEL_DIR}/fold_{f}"
        torch.save(model.state_dict(), path_saved_model)

    return model, np.mean(train_batch_loss), np.mean(test_batch_loss)


# 訓練の実行
epoch = 100
train_loss = []
test_loss = []

for epoch in tqdm(range(epoch)):
    model, train_l, test_l = train_model(model)
    train_loss.append(train_l)
    test_loss.append(test_loss)
    # 10エポックごとにロスを表示
    if epoch % 10 == 0:
        print("Train loss: {a:.3f}, Test loss: {b:.3f}".format(a=train_loss[-1], b = test_loss[-1]))

# 学習状況（ロス）の確認
plt.plot(train_loss, label='train_loss')
plt.plot(test_loss, label='test_loss')
plt.legend()

# 学習済みモデルから予測結果と正解値を取得
def retrieve_result(model, dataloader):
    model.eval()
    preds = []
    labels = []
    # Retreive prediction and labels
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            # Collect data
            preds.append(output)
            labels.append(label)
    # Flatten
    preds = torch.cat(preds, axis=0)
    labels = torch.cat(labels, axis=0)
    # Returns as numpy (CPU環境の場合は不要)
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return preds, labels

# データローダーの作成
test_loader = torch.utils.data.DataLoader(test_dataset,
                                        batch_size=16,
                                        shuffle=False,
                                        num_workers=2,
                                        pin_memory=True,
                                        worker_init_fn=worker_init_fn
                                        )

# 予測結果と正解値を取得
preds, labels = retrieve_result(model, test_loader)

# 学習済みモデルの保存・ロード
# モデルのロード
#model = Mymodel()
#model.load_state_dict(torch.load(path_saved_model))


# Model summary
#from torchsummary import summary
#model = model().to(device)
#summary(model, input_size=(1, 50, 50))