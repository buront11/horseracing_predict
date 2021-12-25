import pickle

import networkx as nx

import dgl

from tqdm import tqdm
import torch
from torch.utils.data import Dataset

import pandas as pd

from sklearn.model_selection import train_test_split

# TODO このままだとtest datasetの読み出しに時間がかかるため後で改善する

class HorceDataset(Dataset):
    def __init__(self, data_type='train'):
        super(HorceDataset, self).__init__()
        # 前処理済みのレースごとの読み込み
        print('loading graph datas...')
        if data_type=='train':
            with open('./train_data', 'rb') as f:
                self.datas = pickle.load(f)
            with open('./train_label', 'rb') as f:
                self.labels = pickle.load(f)
        else:
            with open('./test_data', 'rb') as f:
                self.datas = pickle.load(f)
            with open('./test_label', 'rb') as f:
                self.labels = pickle.load(f)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        out_data = self.datas[idx]
        out_label = self.labels[idx]

        return out_data, out_label

class NodeClassifierDataset(Dataset):
    def __init__(self, data_type='train'):
        super(NodeClassifierDataset, self).__init__()
        # 前処理済みのレースごとの読み込み
        print('loading graph datas...')
        if data_type=='train':
            with open('./train_data', 'rb') as f:
                self.datas = pickle.load(f)
        else:
            with open('./test_data', 'rb') as f:
                self.datas = pickle.load(f)
        print(self.datas[0])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        out_data = self.datas[idx]

        return out_data

class LightGBMDataset(Dataset):
    def __init__(self, dataset):
        super(LightGBMDataset, self).__init__()

        self.dataset = dataset

        self.dataset['label'] = self.dataset['arrival'].apply(lambda x: x if (x==1 or x==2 or x==3) else 0)

        self.datas = self.dataset.drop('label', axis=1)
        self.labels = self.dataset['label']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        out_data = self.datas[idx]
        out_label = self.labels[idx]

        return out_data, out_label

    


if __name__=='__main__':
    dataset = LightGBMDataset()