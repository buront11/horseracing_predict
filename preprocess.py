import pandas as pd
import pickle
import glob
import argparse
import category_encoders as ce
import re
import regex

import json

class Preprocess():
    def __init__(self, args):
        self.horse_db = json.load('./data/horses.json')

        self.race_datas = glob.glob('./data/race_data/*')
        self.race_dfs = [pd.read_csv(str(path)) for path in self.race_datas]

        self.race_df = pd.concat(self.race_dfs, sort=False)
        self._trans_eng_race_col(self.race_df)
        self._split_sex_old(self.race_df)
        self._horse_weight_delta(self.race_df)
        self._time_convert2sec(self.race_df)

        self.graph_datas = []

        # onehot vecに変換する列
        # TODO タイトルを含めるか否か　含めるなら前処理は必要
        # sex, race_type, distance, condition, weather, race_line, rotation, location
        onehot_cols = ['sex','race_type', 'condition', 'weather', 'race_line',
                        'rotation','location']

        self.bine = ce.OneHotEncoder(cols=onehot_cols,handle_unknown='impute')
        self.bine.fit(self.race_df)

        # カテゴリ変数変換する列
        # 馬名　騎手　調教師　馬主
        ordinal_cols = ['horse_name', 'jockey', 'trainer', 'owner']
        self.ce_oe = ce.OrdinalEncoder(cols=ordinal_cols,handle_unknown='impute')
        self.ce_oe.fit(self.race_df)
                    
        self.horse_datas = glob.glob('./data/horse_data/*')
        self.horse_dfs = [pd.read_csv(str(path)) for path in self.horse_datas]

    def _trans_eng_race_col(self, df):
        df = df.rename(columns={'着順':'arrival', '枠番':'number', '馬番':'horse_num', 
                '性齢':'sex_old', '斤量':'weight', '騎手':'jockey', 'タイム':'time', '馬名':'horse_name',
                '着差':'dress_diff','単勝':'win_rate','人気':'popularity', '馬体重':'horse_weight',
                 '調教師':'trainer'})

    def _trans_eng_horse_col(self, df):
        df = df.rename(columns={'着順':'arrival','日付':'day', '開催':'hold', '天気':'weather', 
        'レース名':'race_title', '頭数':'head_num', '枠番':'num', '馬番':'horse_num', 'オッズ':'rate', 
        '人気':'popularity',
       '騎手':'jockey', '斤量':'weight', '距離':'distance', '馬場':'status', 
       'タイム':'time', '着差':'arrival_delta', '通過':'passing',
       'ペース':'pace', '上り':'3f', '馬体重':'horse_weight', '勝ち馬(2着馬)':'second_horse', '賞金':'wining'})

    def _horse_weight_delta(self, df):
        weights = []
        delta_weights = []
        for weight in df['weight']:
            delta_weights.append(int(re.search(r'(?<=\().+?(?=\))', weight).group()))
            weights.append(int(re.sub(r'\(.+\)', '', weight)))
            
        df['weight'] = weights
        df['delta_weight'] = delta_weights

    def _split_sex_old(self, df):
        sex = []
        old = []
        for sex_old in df['sex_old']:
            old.append(re.search(r'[0-9０-９]+', sex_old).group())
            sex.append(re.sub(r'[0-9０-９]+', '', sex_old))

        df['sex'] = sex
        df['old'] = old

    def _time_convert2sec(self, df):
        sec_times = []
        for time in df['time']:
            sec = 0
            times = time.split(':')
            minites = int(times[0])
            for _ in range(minites):
                sec += 60
            sec += float(times[1])
            sec_times.append(sec)
        
        df['sec_time'] = sec_times

    def arrival_converter(arrival):
        kanji = regex.compile(r'\p{Script=Han}+')

        down_flag = 0
        new_arrival = 18
        
        if type(arrival) is int:
            if arrival > 19:
                new_arrival = 18
            else:
                new_arrival = arrival
        else:
            if re.search(r'\(.+\)', str(arrival)):
                new_arrival = int(re.sub(r'\(.+\)', '', str(arrival)))
                down_flag = 1
            elif kanji.search(str(arrival)):
                new_arrival = -1
            else:
                new_arrival = int(arrival)

        return new_arrival, down_flag

    def race_preprocess(self):
        for df in self.race_dfs:
            self._trans_eng_race_col(df)
            self._split_sex_old(df)
            self._horse_weight_delta(df)
            self._time_convert2sec(df)
            df['arrival'], _ = zip(*df['arrival'].apply(self.arrival_converter))

def main(args):
    pp = Preprocess(args)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--race', default='')
    parser.add_argument('--horse', default='')

    args = parser.parse_args()

    main(args)