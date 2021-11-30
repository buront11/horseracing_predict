from logging import raiseExceptions
import os
import re
import json
from tqdm import tqdm
import argparse

import time
from datetime import date, datetime, timedelta
from numpy import not_equal

import pandas as pd 

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

def date_range(start, stop, step = timedelta(1)):
    current = start
    while current < stop:
        yield current
        current += step

def split_race_info(race_info):
    race_info = re.sub(r'\s', '', race_info)
    race_infos = race_info.split('/')
    
    # レース場の種類
    race_types = race_infos[2].split(':')
    race_type = race_types[0]
    # レース場の状態
    race_condition = race_types[1]

    # レース場の天気
    race_weather = race_infos[1].split(':')[1]

    # レースの開始時刻
    race_start_time = re.sub('発走:', '', race_infos[3])

    # レース距離
    race_distance = re.search(r'[0-9０-９]+m', race_infos[0]).group()

    # レース進行
    race_status = re.sub(r'[0-9０-９]+m', '', race_infos[0])
    if len(race_status) == 3:
        race_line = 'out'
        if race_status[1] == '右':
            race_rotation = 'right'
        else:
            race_rotation = 'left'
    else:
        race_line = 'in'
        if race_status[1] == '右':
            race_rotation = 'right'
        else:
            race_rotation = 'left'

    race_infos = [[race_type, race_condition, race_distance, race_weather,
                    race_line, race_rotation, race_start_time]]

    df_race = pd.DataFrame(race_infos,
                            columns=['race_type', 'condition', 'distance', 'weather',
                            'race_line', 'rotation', 'start_time'])

    return df_race

def split_race_detail(race_detail):
    race_details = race_detail.split(' ')

    date = datetime.strptime(race_details[0], "%Y年%m月%d日")
    year = date.year
    month = date.month
    day = date.day

    round = re.search(r'[0-9]+回', race_details[1]).group()
    round = re.sub(r'回', '', round)

    location = re.sub(r'[0-9]+回', '', race_details[1])
    location = re.sub(r'[0-9]+日目', '', location)

    race_title = race_details[2]

    df_race_detail = pd.DataFrame([[year, month, day, location, round, race_title]],
                            columns=['year', 'month', 'day', 'location', 'round', 'title'])

    return df_race_detail
    
def repeat_df(df, num):
    origin_df = df
    for _ in range(num-1):
        df = df.append(origin_df)

    return df.reset_index()

class HorceDateCollecter():
    def __init__(self):
        if not os.path.isdir('./data/'):
            os.makedirs('./data/')
            self.horse_ids = {}
            self.jockey_ids = {}
            self.trainer_ids = {}
            self.owner_ids = {}
        else:
            # 馬のidを取得
            if not os.path.isfile('./data/horses.json'):
                self.horse_ids = {}
            else:
                with open('./data/horses.json', 'r') as f:
                    self.horse_ids = json.load(f)
            # 騎手のidを取得
            if not os.path.isfile('./data/jockeys.json'):
                self.jockey_ids = {}
            else:
                with open('./data/jockeys.json', 'r') as f:
                    self.jockey_ids = json.load(f)
            # 調教師のidを取得
            if not os.path.isfile('./data/trainers.json'):
                self.trainer_ids = {}
            else:
                with open('./data/trainers.json', 'r') as f:
                    self.trainer_ids = json.load(f)
            # 所有者のidを取得
            if not os.path.isfile('./data/owners.json'):
                self.owner_ids = {}
            else:
                with open('./data/owners.json', 'r') as f:
                    self.owner_ids = json.load(f)

        if not os.path.isdir('./data/race_data'):
            os.makedirs('./data/race_data/')

        if not os.path.isdir('./data/horse_data'):
            os.makedirs('./data/horse_data/')

        if not os.path.isdir('./data/pedigree_data'):
            os.makedirs('./data/pedigree_data/')

        # 前回の実行時刻を取得
        try:
            with open('last_updata_log.txt', 'r') as f:
                self.start_date = datetime.strptime(f.read(), '%Y-%m-%d').date()
        except FileNotFoundError:
            # 前回のデータがない場合は1986年1月1日からスクレイピング
            self.start_date = datetime(1986, 1, 1).date()
            

        # 前回のデータとの差分をとって最新のデータのみ持ってくる
        self.dt_today = datetime.utcnow().date()

    def get_race_data(self):
        for date in date_range(self.start_date, self.dt_today):
            # データをとった日付を記録
            with open('last_updata_log.txt', 'w') as f:
                f.write(str(date))

            date = re.sub('-', '', str(date))
            url = 'https://db.netkeiba.com/race/list/' + date
            res = requests.get(url)

            soup = BeautifulSoup(res.content, 'html.parser')
            hold_race = soup.select_one('div.race_kaisai_info')
            if hold_race is None:
                continue

            print('get date {} now ...'.format(date))

            race_urls = hold_race.find_all('a', title=re.compile(r'.+'))
            for index in tqdm(range(len(race_urls))):
                race = race_urls[index].get('href')
                url = 'https://db.netkeiba.com' + race
                race_id = re.search(r'[0-9]+', race).group()

                res = requests.get(url)
                soup = BeautifulSoup(res.content, 'html.parser')

                # 出走馬情報
                df_horses = pd.read_html(url, match='馬名')[0]
                info_ids = soup.select('td.txt_l a')
                for info in info_ids:
                    info_type = re.search(r'[a-z]+', info.get('href')).group()
                    name = info.get_text()
                    id = re.search(r'[0-9]+', info.get('href')).group()

                    if info_type == 'horse':
                        if id not in self.horse_ids:
                            self.horse_ids.update({name:id})
                    elif info_type == 'jockey':
                        if id not in self.jockey_ids:
                            self.jockey_ids.update({name:id})
                    elif info_type == 'trainer':
                        if id not in self.trainer_ids:
                            self.trainer_ids.update({name:id})
                    elif info_type == 'owner':
                        if id not in self.owner_ids:
                            self.owner_ids.update({name:id})

                # レース情報　馬場距離/天候/状態/発走時刻
                race_info = soup.select_one('div.data_intro diary_snap_cut span').get_text()
                df_race_info = split_race_info(race_info)
                df_race_info = repeat_df(df_race_info, len(df_horses))
                # レース詳細 日付/回数/レース名/出走馬種
                race_detail = soup.select_one('p.smalltxt').get_text()
                df_race_detail = split_race_detail(race_detail)
                df_race_detail = repeat_df(df_race_detail, len(df_horses))

                df_race = pd.concat([df_horses, df_race_info, df_race_detail], axis=1)

                df_race.to_csv('./data/race_data/' + race_id + '.csv', index=False)

                time.sleep(2)

            # 各日付ごとに逐一保存
            self._save_ids()

    def get_horse_date(self):
        # 馬の情報の取得
        for horse_name, horse_id in self.horse_ids.items():
            print('get {} datas ...'.format(horse_name))

            url = 'https://db.netkeiba.com/horse/' + horse_id

            res = requests.get(url)
            soup = BeautifulSoup(res.content, 'html.parser')

            df_horse = pd.read_html(url, match='レース名')[0]

            df_horse.to_csv('./data/horse_data/' + horse_id + '.csv', index=False)

            time.sleep(2)

    def _save_ids(self):
        # 馬のidを保存
        with open('./data/horses.json', 'w') as f:
            json.dump(self.horse_ids, f, ensure_ascii=False, indent=2)
        # 騎手のidを保存
        with open('./data/jockeys.json', 'w') as f:
            json.dump(self.jockey_ids, f, ensure_ascii=False, indent=2)
        # 調教師のidを保存
        with open('./data/trainers.json', 'w') as f:
            json.dump(self.trainer_ids, f, ensure_ascii=False, indent=2)
        # 所有者のidを保存
        with open('./data/owners.json', 'w') as f:
            json.dump(self.owner_ids, f, ensure_ascii=False, indent=2)

def main(args):

    horce_collect = HorceDateCollecter()

    horce_collect.get_race_data()

    horce_collect.get_horse_date()
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--http_proxy', default='')
    parser.add_argument('--https_proxy', default='')

    args = parser.parse_args()

    main(args)
