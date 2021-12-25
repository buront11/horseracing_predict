import pickle

import numpy as np

import torch
import torch.optim as optim 
import torch.nn as nn
from torch.cuda import device

from train import train, node_train
from models import GATClassifier, NodeClassifier
from datasets import HorceDataset, NodeClassifierDataset, LightGBMDataset

import pandas as pd
from sklearn.model_selection import train_test_split

import lightgbm as lgb

def main():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    train_dataset = NodeClassifierDataset('train')
    # test_dataset = HorceDataset('test')
    with open('./test_data', 'rb') as f:
        datas = pickle.load(f)
    with open('./test_label', 'rb') as f:
        labels = pickle.load(f)

    model = NodeClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    print(train_dataset[0])

    node_train(train_dataset, model, criterion, optimizer, batch_size=64, epochs=20, device=device)

def light_gbm_main():

    dataset = pd.read_pickle('./lightgbm_dataset.pickle')

    df = dataset

    X = dataset.drop('label', axis=1).values
    y = dataset['label'].values

    X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.9, random_state=123)

    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # テストデータのクラス予測確率 (各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率] を返す)
    y_pred_prob = model.predict_proba(X_test)

    pd.set_option('display.max_rows', 500)

    df_pred = pd.DataFrame({'target':y_test,'target_pred':y_pred})
    print(df_pred.head(500))

    # 真値と予測確率の表示
    df_pred_prob = pd.DataFrame({'target':y_test, 'target0_prob':y_pred_prob[:,0], 'target1_prob':y_pred_prob[:,1]})
    print(df_pred_prob.head(500))

    cols = list(df.drop('label',axis=1).columns)       # 特徴量名のリスト(目的変数CRIM以外)
    # 特徴量重要度の算出方法 'gain'(推奨) : トレーニングデータの損失の減少量を評価
    f_importance = model.feature_importances_ # 特徴量重要度の算出 //
    # f_importance = f_importance / np.sum(f_importance) # 正規化(必要ない場合はコメントアウト)
    df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
    df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
    print(df_importance)

if __name__=='__main__':
    light_gbm_main()