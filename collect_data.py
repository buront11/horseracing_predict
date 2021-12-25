from logging import raiseExceptions
import os
import re
import regex
import glob
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

def split_race_info(race_info, race_title, race_round):
    race_info = re.sub(r'\s', '', race_info)
    race_infos = race_info.split('/')

    race_round = int(re.sub(r'\s+R', '', race_round))
    
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
        race_rotation= race_status[1]
    else:
        race_line = 'in'
        race_rotation = race_status[1]

    race_infos = [[race_type, race_condition, race_distance, race_weather,
                    race_line, race_rotation, race_start_time, race_title, race_round]]

    df_race = pd.DataFrame(race_infos,
                            columns=['race_type', 'condition', 'distance', 'weather',
                            'race_line', 'rotation', 'start_time', 'title', 'round'])

    return df_race

def split_race_detail(race_detail):
    race_details = race_detail.split(' ')

    date = datetime.strptime(race_details[0], "%Y年%m月%d日")
    year = date.year
    month = date.month
    day = date.day

    race_num = re.search(r'[0-9]+回', race_details[1]).group()
    race_num = re.sub(r'回', '', race_num)

    location = re.sub(r'[0-9]+回', '', race_details[1])
    location = re.sub(r'[0-9]+日目', '', location)

    race_names = race_details[2].split()
    race_title = race_names[0]

    df_race_detail = pd.DataFrame([[date, year, month, day, location, race_num, race_title]],
                            columns=['date', 'year', 'month', 'day', 'location', 'race_num', 'sub_title'])

    return df_race_detail
    
def repeat_df(df, num):
    origin_df = df
    for _ in range(num-1):
        df = df.append(origin_df)

    return df.reset_index(drop=True)

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
            # 前回のデータがない場合は2008年1月1日からスクレイピング
            self.start_date = datetime(2008, 1, 1).date()
            
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
            res.encoding = res.apparent_encoding

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
                res.encoding = res.apparent_encoding

                soup = BeautifulSoup(res.content, 'html.parser')

                # 出走馬情報
                df_horses = pd.read_html(res, match='馬名')[0]
                info_ids = soup.select('td.txt_l a')
                for info in info_ids:
                    info_type = re.search(r'[a-z]+', info.get('href')).group()
                    name = info.get_text()
                    id = re.search(r'[0-9]+', info.get('href')).group()

                    if info_type == 'horse':
                        if id not in self.horse_ids.values():
                            self.horse_ids.update({name:id})
                        df_horses['horse_id'] = id
                    elif info_type == 'jockey':
                        if id not in self.jockey_ids.values():
                            self.jockey_ids.update({name:id})
                        df_horses['jockey_id'] = id
                    elif info_type == 'trainer':
                        if id not in self.trainer_ids.values():
                            self.trainer_ids.update({name:id})
                        df_horses['trainer_id'] = id
                    elif info_type == 'owner':
                        if id not in self.owner_ids.values():
                            self.owner_ids.update({name:id})
                        df_horses['owner_id'] = id

                # レース情報　レース名/馬場距離/天候/状態/発走時刻
                race_title = soup.select_one('div.data_intro h1').get_text()
                race_round = soup.select_one('div.data_intro dl.racedata.fc dt').get_text()
                race_info = soup.select_one('div.data_intro diary_snap_cut span').get_text()
                df_race_info = split_race_info(race_info, race_title, race_round)
                df_race_info = repeat_df(df_race_info, len(df_horses))
                # レース詳細 日付/回数/サブレース名/出走馬種
                try:
                    race_detail = soup.select_one('p.smalltxt').get_text()
                    df_race_detail = split_race_detail(race_detail)
                    df_race_detail = repeat_df(df_race_detail, len(df_horses))

                    df_race = pd.concat([df_horses, df_race_info, df_race_detail], axis=1)

                    df_race.to_csv('./data/race_data/' + race_id + '.csv', index=False)
                except:
                    print('broken race data | Race ID:{}'.format(race_id))
                    continue

                time.sleep(1)

            # 各日付ごとに逐一保存
            self._save_ids()

    def get_horse_date(self):
        # 馬の情報の取得
        for horse_name, horse_id in self.horse_ids.items():
            print('get {} datas ...'.format(horse_name))

            url = 'https://db.netkeiba.com/horse/' + horse_id

            res = requests.get(url)
            res.encoding = res.apparent_encoding
            soup = BeautifulSoup(res.content, 'html.parser')

            df_horse = pd.read_html(res, match='レース名')[0]

            df_horse.to_csv('./data/horse_data/' + horse_id + '.csv', index=False)

            time.sleep(1)

    def get_correct_jockey_name(self):
        new_jockey_ids = {} 

        for jockey, id in self.jockey_ids.items():
            print('get {} datas ...'.format(jockey))

            url = 'https://db.netkeiba.com/jockey/' + id

            res = requests.get(url)
            res.encoding = res.apparent_encoding
            soup = BeautifulSoup(res.content, 'html.parser')

            jockey_name = soup.select_one('div.db_head_name.fc h1').get_text()
            jockey_name = re.sub(r'\n', '', jockey_name)
            jockey_name = re.sub(r'\(.+\)', '', jockey_name)
            jockey_name = re.sub(r'\s', '', jockey_name)

            if jockey_name not in new_jockey_ids.keys():
                new_jockey_ids.update({jockey_name:id})

            time.sleep(1)

        self.jockey_ids = new_jockey_ids
        with open('./data/jockeys.json', 'w') as f:
            json.dump(self.jockey_ids, f, ensure_ascii=False, indent=2)

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

    def _fix_garbled_dict(self, dict):
        """文字化けしているデータを削除する
            文字化けしているデータはなぜか重複データなので問答無用で消して問題ない

        Parameters
        ----------
        dict : 
            [消したい辞書]
        """
        hiragana = re.compile('[\u3041-\u309F]+')
        katakana = re.compile('[\u30A1-\u30FF]+')
        kanji = regex.compile(r'\p{Script=Han}+')

        del_keys = []

        for key in dict.keys():
            if katakana.search(key):
                continue
            elif kanji.search(key):
                continue
            elif hiragana.search(key):
                continue
            else:
                del_keys.append(key)

        for key in del_keys:
            del dict[key]

    def fix_garbled_char(self):
        self._fix_garbled_dict(self.horse_ids)
        self._fix_garbled_dict(self.jockey_ids)
        self._fix_garbled_dict(self.trainer_ids)
        self._fix_garbled_dict(self.owner_ids)

        self._save_ids()

def complement_horse_name():
    """現状馬のデータをスクレイピングしても名前がjsonに登録されない馬がいるので
        それを補完するプログラム
    """
    with open('./data/horses.json') as f:
        horse_db = json.load(f)
    horse_csvs = glob.glob('./data/horse_data/*')
    horse_ids = [os.path.splitext(os.path.basename(path))[0] for path in horse_csvs]
    katakana = re.compile('[\u30A1-\u30FF]+')
    
    for i in tqdm(range(len(horse_ids))):
        horse_name = [k for k, v in horse_db.items() if v == horse_ids[i]][0]
        if katakana.search(horse_name):
            continue
        del horse_db[horse_name]
        url = 'https://db.netkeiba.com/horse/' + str(horse_ids[i])
        res = requests.get(url)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.content, 'html.parser')
        horse_name = soup.select_one('div.db_head_name.fc div.horse_title h1').get_text()
        horse_name = katakana.search(horse_name).group()
        print(horse_name)
        # 空白の削除
        horse_name = re.sub(r'\s', '', horse_name)
        time.sleep(1)
        if horse_name not in horse_db.keys():
            horse_db.update({horse_name:horse_ids[i]})

    with open('./data/horses.json', 'w') as f:
            json.dump(horse_db, f, ensure_ascii=False, indent=2)
        
def main(args):

    # horce_collect = HorceDateCollecter()

    # horce_collect.get_race_data()

    # horce_collect.get_horse_date()

    # horce_collect.get_correct_jockey_name()

    # horce_collect.fix_garbled_char()

    complement_horse_name()
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--http_proxy', default='')
    parser.add_argument('--https_proxy', default='')

    args = parser.parse_args()

    main(args)
