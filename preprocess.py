import pandas as pd
import pickle
import glob
import argparse
import category_encoders as ce
import re
import os
import requests
from bs4 import BeautifulSoup
import regex
import math
import numpy as np
from datetime import datetime
from fractions import Fraction

from tqdm import tqdm

import json

class Preprocess():
    def __init__(self, args):
        with open('./data/horses.json') as f:
            self.horse_db = json.load(f)

        if args.horse or not(os.path.isfile('./preprocessed_horse_df.pickle')):
            print('loading horse datas...')
            self.horse_datas = glob.glob('./data/horse_data/*')
            self.horse_dfs = {os.path.splitext(os.path.basename(self.horse_datas[i]))[0]:\
                pd.read_csv(str(self.horse_datas[i])) for i in tqdm(range(len(self.horse_datas)))}
            # データが膨大なため早めにdelする
            del self.horse_datas
            self.horse_preprocess()
        else:
            print('loading horse datas...')
            with open('./preprocessed_horse_df.pickle', 'rb') as f:
                self.horse_dfs = pickle.load(f)

        self.race_datas = glob.glob('./data/race_data/*')
        print('loading race datas...')
        self.race_dfs = [pd.read_csv(str(self.race_datas[i])) for i in tqdm(range(len(self.race_datas)))]

        self.race_df = pd.concat(self.race_dfs, sort=False)
        self.race_df = self._trans_eng_race_col(self.race_df)
        self.race_df = self._split_sex_old(self.race_df)
        # self.race_df = self._time_convert2sec(self.race_df)

        self.graph_datas = []

        # onehot vecに変換する列
        # TODO タイトルを含めるか否か　含めるなら前処理は必要
        # sex, race_type, distance, condition, distance, weather, race_line, rotation, location, sub_title
        self.onehot_cols = ['sex','race_type', 'condition', 'weather', 'race_line','distance',
                        'rotation','location', 'sub_title']

        self.bine = ce.OneHotEncoder(cols=self.onehot_cols,handle_unknown='impute')
        self.bine.fit(self.race_df.loc[:,self.onehot_cols])

        # カテゴリ変数変換する列
        # 馬名　騎手　調教師　x馬主:データ取れてない
        self.ordinal_cols = ['horse_name', 'jockey', 'trainer']
        self.ce_oe = ce.OrdinalEncoder(cols=self.ordinal_cols,handle_unknown='impute')
        self.ce_oe.fit(self.race_df.loc[:,self.ordinal_cols])

    def _trans_eng_race_col(self, df):
        # レースのdfの列名を英語に変換
        df = df.rename(columns={'着順':'arrival', '枠番':'number', '馬番':'horse_num', 
                '性齢':'sex_old', '斤量':'weight', '騎手':'jockey', 'タイム':'time', '馬名':'horse_name',
                '着差':'arrival_diff','単勝':'win_rate','人気':'popularity', '馬体重':'horse_weight',
                 '調教師':'trainer'})

        return df

    def _trans_eng_horse_col(self, df):
        # 馬のdfの列名を英語に変換
        df = df.rename(columns={'着順':'arrival','日付':'date', '開催':'hold', '天気':'weather', 
            'レース名':'race_title', '頭数':'head_num', '枠番':'number', '馬番':'horse_num', 'オッズ':'rate', 
            '人気':'popularity','R':'round',
            '騎手':'jockey', '斤量':'weight', '距離':'distance', '馬場':'condition', 
            'タイム':'time', '着差':'arrival_diff', '通過':'passing',
            'ペース':'pace', '上り':'3f', '馬体重':'horse_weight', '勝ち馬(2着馬)':'second_horse', '賞金':'winning'})
        return df

    def _horse_weight_delta(self, df):
        # 体重を増加減少と体重のみの情報に分割
        weights = []
        delta_weights = []
        for weight in df['horse_weight']:
            if weight == '計不':
                weights.append(np.nan)
                delta_weights.append(0)
            else:
                if re.search(r'(?<=\().+?(?=\))', weight):
                    delta_weights.append(int(re.search(r'(?<=\().+?(?=\))', weight).group()))
                    weights.append(int(re.sub(r'\(.+\)', '', weight)))
                else:
                    delta_weights.append(0)
                    weights.append(int(weight))
            
        df['horse_weight'] = weights
        df['delta_horse_weight'] = delta_weights

        return df

    def _split_sex_old(self, df):
        # 性齢を性別と年齢に分割
        sex = []
        old = []
        for sex_old in df['sex_old']:
            old.append(re.search(r'[0-9０-９]+', sex_old).group())
            sex.append(re.sub(r'[0-9０-９]+', '', sex_old))

        df['sex'] = sex
        df['old'] = old

        return df

    def _time_convert2sec(self, df):
        # レースのタイムを秒数に変換
        sec_times = []
        for time in df['time']:
            if type(time) is str:
                sec = 0
                if re.search(r'短頭', time):
                    sec_times.append(np.nan)
                elif re.search(r':', time):
                    times = time.split(':')
                    minites = int(times[0])
                    for _ in range(minites):
                        sec += 60
                    sec += float(times[1])
                    sec_times.append(sec)
                elif re.search(r'.', time):
                    times = time.split('.')
                    minites = int(times[0])
                    for _ in range(minites):
                        sec += 60
                    sec += float(times[1])
                    sec += float(times[2])/10
                    sec_times.append(sec)
            else:
                if math.isnan(time):
                    sec_times.append(time)
                else:
                    print(time)
                    exit(1)
        
        df['sec_time'] = sec_times

        return df

    def _split_harf_pace(self, df):
        # ペースを前半と後半に分割する
        up_pace = []
        down_pace = []
        for pace in df['pace']:
            if type(pace) is str:
                paces = [float(i) for i in pace.split('-')]
                up_pace.append(paces[0])
                down_pace.append(paces[1])
            else:
                if math.isnan(pace):
                    up_pace.append(pace)
                    down_pace.append(pace)
                else:
                    print('pace')
                    exit(1)

        df['up_pace'] = up_pace
        df['up_pace'] = df['up_pace'].fillna(df['up_pace'].mean())
        df['down_pace'] = down_pace
        df['down_pace'] = df['down_pace'].fillna(df['down_pace'].mean())

        return df

    def _get_race_rank(self, df):
        # レースの格によってランク付け
        ranks = []
        for race_title in df['race_title']:
            race_rank = 0
            if re.search(r'(未勝利|新馬)', race_title):
                race_rank = -1
            elif re.search(r'G1', race_title):
                race_rank = 3
            elif re.search(r'G2', race_title):
                race_rank = 2
            elif re.search(r'G3', race_title):
                race_rank = 1
            ranks.append(race_rank)

        df['rank'] = ranks

        return df

    def _convert_arrival_diff(self, df):
        diffs = []
        for diff in df['arrival_diff']:
            diff = str(diff)
            if diff == 'ハナ':
                diffs.append(0.2)
            elif diff == 'アタマ':
                diffs.append(0.4)
            elif diff == 'クビ':
                diffs.append(0.8)
            elif re.search(r'.', diff):
                diff_dis = diff.split('.')
                int_num = int(diff_dis[0])
                float_num = float(Fraction(diff_dis[1]))
                diffs.append(int_num*2.4+float_num*2.4)
            else:
                int_num = int(diff)*2.4

        df['arrival_diff'] = diffs

        return df

    def _fix_race_nan(self, df):
        pass

    def _fix_horse_nan(self, df):
        # 馬のnan値の処理
        # 未獲得の賞金は0で埋める
        # 基本枠番がnanのやつは出場できていないのでdropする
        df['winning'] = df['winning'].fillna(0)
        df['rate'] = df['rate'].fillna(df['rate'].mean())
        df['popularity'] = df['popularity'].fillna(df['popularity'].mean().astype(int))
        df['weight'] = df['weight'].fillna(df['weight'].mean().astype(int))
        # 秒数に変換してからnanを処理する
        df = self._time_convert2sec(df)
        df['sec_time'] = df['sec_time'].fillna(df['sec_time'].mean())
        df['arrival_diff'] = df['arrival_diff'].fillna(0)
        # 馬体重の分割後、計測不可を処理
        df = self._horse_weight_delta(df)
        # 海外が主で体重のデータがない馬は競走馬の平均体重(らしい)470とする
        try:
            df['horse_weight'] = df['horse_weight'].fillna(df['horse_weight'].mean().astype(int))
        except:
            df['horse_weight'] = df['horse_weight'].fillna(470)
        # 失格は最下位以下の順位とする
        df['arrival'] = df['arrival'].fillna(19)

        return df

    def _get_horse_data(self, df):
        # 対応する馬のデータを持ってくるプログラム　あとでかく
        # 日付をdatetime型に変換
        df['date'] = df['date'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))
        horse_datas = []
        for row in df.itertuples():
            # 馬のidを取得
            horse_id = self.horse_db[row.horse_name]

            race_date = row.date
            horse_df = self.horse_dfs[horse_id]
            # 日付の同じ馬のデータを抽出する
            horse_datas.append(horse_df[horse_df['date'] == race_date])

        add_df = pd.concat(horse_datas, sort=False)
        try:
            # なぜかweatherとdistanceが消されない場合があるので明示的に削除する
            add_df = add_df.drop(['weather', 'distance'], axis=1)
        except:
            pass

        df = pd.merge(df, add_df, how='outer',\
             on=['date', 'jockey', 'weight', 'condition','round', 'number', 'horse_num', 'popularity']
                 ).reset_index(drop=True)

        return df

    def arrival_converter(self, df):
        new_arrivals = []
        down_flags = []

        for arrival in df['arrival']:
            # 着順の表記の乱れを修正する
            # 着順がnanもしくは失などの場合、失格として最下位よりも下の19位とする(脚質の際に重みの考慮がなくなる)
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
                    new_arrival = 19
                else:
                    new_arrival = int(arrival)
            new_arrivals.append(new_arrival)
            down_flags.append(down_flag)

        return new_arrivals, down_flags

    def get_leg_quality(self, df):
        # 脚質を数値として導出
        race = []
        weight = []
        for row in df[(~(df['passing'].isnull()))&(~(df['head_num'] == 0))&(~(df['head_num'].isnull()))&(~(df['arrival'] == 19))].itertuples():
            passing_result = [int(x) for x in row.passing.split('-')]
            mean_val = sum(passing_result)/len(passing_result)
            w = abs((int(row.head_num) - int(row.arrival) + 1)/int(row.head_num))
            race.append(mean_val)
            weight.append(w)
        
        # 一つも通過の値がない場合中間の5とする
        if len(weight) == 0:
            df['leg'] = 5
        else:
            race = np.array(race)
            weight = np.array(weight)
            
            df['leg'] = np.average(race, weights=weight)

        return df

    def potential_distance(df):
        pass
    # 距離適性を求めるプログラム　いつか作る

    def _last_race_day(self, df):
        # 最後にレースを行ってからの日付を取得する
        df['date'] = df['date'].apply(lambda x:datetime.strptime(x, '%Y/%m/%d'))
        last_day = df['date'].diff().apply(lambda x: 0 if pd.isnull(x) else x.days)

        last_days = []

        for day in reversed(last_day):
            last_days.append(abs(day))
        
        df['last_day'] = last_days

        return df

    def _get_total_winning(self, df):
        # 賞金の獲得総額を取得する
        df['winning'] = df['winning'].fillna(0)
        total_winning = 0
        total_winnings = []
        for money in reversed(df['winning']):
            total_winning += money
            total_winnings.append(total_winning)

        total_winnings = reversed(total_winnings)
        df['total_winning'] = list(total_winnings)

        return df

    def race_preprocess(self):
        print('preprocessing race datas...')
        for i in tqdm(range(len(self.race_dfs))):
            # 列名を英語に変換
            self.race_dfs[i] = self._trans_eng_race_col(self.race_dfs[i])
            # 性別と年齢の追加(列数を合わせるため)
            self.race_dfs[i] = self._split_sex_old(self.race_dfs[i])
            # 不要な列(馬のデータの方に存在しており前処理が不必要なデータ)を削除
            # 着差とタイム
            self.race_dfs[i] = self.race_dfs[i].drop(['arrival','time','arrival_diff',\
                'horse_weight'], axis=1)
            # レースの情報以外は馬のデータから取ってくる
            # idは不要だったので消す
            self.race_dfs[i] = self.race_dfs[i].drop(['horse_id','jockey_id','trainer_id','owner_id'], axis=1)
            # 馬のデータを結合する
            self.race_dfs[i] = self._get_horse_data(self.race_dfs[i])
            df = self.race_dfs[i]
            # カテゴリ変数をonehot encoding
            self.race_dfs[i] = pd.concat([self.race_dfs[i], \
                self.bine.transform(self.race_dfs[i].loc[:,self.onehot_cols])], axis=1)
            # onehot encodingした元のデータを削除
            self.race_dfs[i] = self.race_dfs[i].drop(self.onehot_cols, axis=1)
            # 大規模なカーディナリデータをordirnaly encoding
            self.race_dfs[i] = pd.concat([self.race_dfs[i], \
                self.ce_oe.transform(self.race_dfs[i].loc[:,self.ordinal_cols])], axis=1)
            # 失格(着順が19のデータ)は削除
            self.race_dfs[i] = self.race_dfs[i][~(self.race_dfs[i]['arrival'] == 19)]
            
        with open('preprocessed_race_data.pickle', 'wb') as f:
            pickle.dump(self.race_dfs, f)
            
    def horse_preprocess(self):
        print('preprocessing horse datas...')
        for key in tqdm(self.horse_dfs.keys()):
            # 最初に列名を英語に変換する
            self.horse_dfs[key] = self._trans_eng_horse_col(self.horse_dfs[key])
            if len(self.horse_dfs[key]) <= 1:
                continue
            # nan値の補完
            self.horse_dfs[key] = self._fix_horse_nan(self.horse_dfs[key])
            # 着差の数値変換
            self.horse_dfs[key] = self._convert_arrival_diff(self.horse_dfs[key])
            # 着順の変換と降格フラグの追加
            arrival_list, down_flag_list = self.arrival_converter(self.horse_dfs[key])
            self.horse_dfs[key]['arrival'] = arrival_list
            self.horse_dfs[key]['down_flag'] = down_flag_list
            # ペース値の新たな列の追加
            self.horse_dfs[key] = self._split_harf_pace(self.horse_dfs[key])
            # レースランクの追加
            self.horse_dfs[key] = self._get_race_rank(self.horse_dfs[key])
            # 前回レース日の追加
            self.horse_dfs[key] = self._last_race_day(self.horse_dfs[key])
            # 賞金総額の追加
            self.horse_dfs[key] = self._get_total_winning(self.horse_dfs[key])
            # 脚質の追加
            self.horse_dfs[key] = self.get_leg_quality(self.horse_dfs[key])
            # 不要な列を削除する この処理は過去のレース情報を含めない場合のため、含める場合は処理を変えること
            drop_cols = ['hold', 'weather', 'race_title', '映像', 'distance', \
                '馬場指数', 'ﾀｲﾑ指数', '厩舎ｺﾒﾝﾄ', '備考', 'second_horse', 'winning']
            self.horse_dfs[key] = self.horse_dfs[key].drop(drop_cols, axis=1)

        with open('preprocessed_horse_df.pickle', 'wb') as f:
            pickle.dump(self.horse_dfs, f)

def main(args):
    pp = Preprocess(args)

    pp.race_preprocess()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--race', action='store_true')
    parser.add_argument('--horse', action='store_true')

    args = parser.parse_args()

    main(args)