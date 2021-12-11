import pickle

import networkx as nx

import dgl

import torch
from torch.utils.data import Dataset

class HorceDataset(Dataset):
    def __init__(self, data_type='train'):
        super(HorceDataset, self).__init__()
        # 前処理済みのレースごとの読み込み
        with open('preprocessed_race_data.pickle', 'rb') as f:
            dfs = pickle.load(f)

        self.train_graph_datas = []
        self.train_graph_labels = []

        self.test_graph_datas = []
        self.test_graph_labels = []

        for df in dfs:
            self.df2graph()

        if data_type == 'train':
            self.datas = self.train_graph_datas
            self.labels = self.train_graph_labels
        else:
            self.datas = self.test_graph_datas
            self.labels = self.test_graph_labels

    def df2graph(self, df):
        # 全てのdfの時間は共通
        if df['year'][0] == '2021':
            test_flag = True
        else:
            test_flag = False

        # デフォルトだと一番上の要素が着順で1位になるため、dfの行をシャッフルする
        df = df.sample(frac=1)

        # 一位の行を取得
        label_index = df.index[df['arrival'] == 1].tolist()[0]

        drop_cols = ['sex_old', 'start_time','title', 'date', 'year', 'month', 'day',\
                    'time', 'passing', 'pace', 'down_flag', 'up_pace', 'down_pace', 'arrival',\
                    'horse_name', 'jockey', 'trainer']

        df = df.drop(drop_cols, axis=1)

        # 年齢がstrだったのでintに変換
        # TODO 前処理の段階で直すようにする
        df['old'] = df['old'].astype(int)
        
        # 出馬している馬の頭数
        horse_num = len(df)
        # 出馬している馬の頭数分の完全グラフを作成する
        graph = nx.complete_graph(horse_num)

        for index, row in enumerate(df.values.tolist()):
            graph.nodes[index]['feat'] = row

        dgl_graph = dgl.from_networkx(graph, node_attrs=['feat'], device='cuda')
        if test_flag:
            self.test_graph_datas.append(dgl_graph)
            self.test_graph_labels.append(label_index)
        else:
            self.test_graph_datas.append(dgl_graph)
            self.test_graph_labels.append(label_index)

    def __getitem__(self, idx):
        out_data = self.datas[idx]
        out_label = self.labels[idx]

        return out_data, out_label

if __name__=='__main__':
    HorceDataset()