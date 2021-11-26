from logging import raiseExceptions
import os
import re
import json
import argparse

import time
from datetime import date, datetime, timedelta

import pandas as pd 

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

def date_range(start, stop, step = timedelta(1)):
    current = start
    while current < stop:
        yield re.sub('-', '', str(current))
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

class ChromeDriver():
    def __init__(self):
        options = webdriver.ChromeOptions()
        self.driver = webdriver.Chrome(ChromeDriverManager().install())

        # 前回のデータとの差分をとって最新のデータのみ持ってくる
        self.dt_today = datetime.datetime.utcnow().date()

    def getdriver(self):
        return self.driver

def main(args):
    HTTP_PROXY = args.http_proxy
    HTTPS_RPOXY = args.https_proxy

    proxies = {
        "http":HTTP_PROXY,
        "https":HTTPS_RPOXY
    }

    # options = webdriver.ChromeOptions()
    # driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    
    if not os.path.isdir('./data/'):
        os.makedirs('./data/')
        horse_ids = {}
        jockey_ids = {}
        trainer_ids = {}
        owner_ids = {}
    else:
        # 馬のidを取得
        if not os.path.isfile('./data/horses.json'):
            horse_ids = {}
        else:
            with open('./data/horses.json', 'r') as f:
                horse_ids = json.load(f)
        # 騎手のidを取得
        if not os.path.isfile('./data/jockeys.json'):
            jockey_ids = {}
        else:
            with open('./data/jockeys.json', 'r') as f:
                jockey_ids = json.load(f)
        # 調教師のidを取得
        if not os.path.isfile('./data/trainers.json'):
            trainer_ids = {}
        else:
            with open('./data/trainers.json', 'r') as f:
                trainer_ids = json.load(f)
        # 所有者のidを取得
        if not os.path.isfile('./data/owners.json'):
            owner_ids = {}
        else:
            with open('./data/owners.json', 'r') as f:
                owner_ids = json.load(f)

    if not os.path.isdir('./data/race_data'):
        os.makedirs('./data/race_data/')

    if not os.path.isdir('./data/horse_data'):
        os.makedirs('./data/horse_data/')

    if not os.path.isdir('./data/pedigree_data'):
        os.makedirs('./data/pedigree_data/')

    # 前回の実行時刻を取得
    try:
        with open('last_updata_log.txt', 'r') as f:
            start_date = f.read()
    except FileNotFoundError:
        # 前回のデータがない場合は1986年1月1日からスクレイピング
        start_date = datetime(1986, 1, 1).date()

    # 実行時の日付までのデータを取得
    dt_today = datetime.utcnow().date()

    for date in date_range(start_date, dt_today):
        url = 'https://db.netkeiba.com/race/list/' + date
        res = requests.get(url)

        soup = BeautifulSoup(res.content, 'html.parser')
        hold_race = soup.select_one('div.race_kaisai_info')
        if hold_race is None:
            continue

        print('get date {} now ...'.format(date))

        race_urls = hold_race.find_all('a', title=re.compile(r'.+'))
        for race in race_urls:
            race = race.get('href')
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
                    if id not in horse_ids:
                        horse_ids.update({id:name})
                elif info_type == 'jockey':
                    if id not in jockey_ids:
                        jockey_ids.update({id:name})
                elif info_type == 'trainer':
                    if id not in trainer_ids:
                        trainer_ids.update({id:name})
                elif info_type == 'owner':
                    if id not in owner_ids:
                        owner_ids.update({id:name})

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

            time.sleep(3)

    # 馬の情報の取得
    for horse_id in horse_ids.values():
        url = 'https://db.netkeiba.com/horse/' + horse_id

        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')

        df_horse = pd.read_html(url, match='レース名')[0]

        df_horse.to_csv('./data/horse_data/' + horse_id + '.csv', index=False)

        time.sleep(3)

    # 馬の血統の取得
    # めんどくさいので後にする
    time.sleep(1)

    # driver.get(url)

    # driver.quit()

    # 実行した日付を記録
    with open('last_updata_log.txt', 'w') as f:
        f.write(str(dt_today))

    # 馬のidを保存
    with open('./data/horses.json', 'w') as f:
        json.dump(horse_ids, f, ensure_ascii=False)
    # 騎手のidを保存
    with open('./data/jockeys.json', 'w') as f:
        json.dump(jockey_ids, f, ensure_ascii=False)
    # 調教師のidを保存
    with open('./data/trainers.json', 'w') as f:
        json.dump(trainer_ids, f, ensure_ascii=False)
    # 所有者のidを保存
    with open('./data/owners.json', 'w') as f:
        json.dump(owner_ids, f, ensure_ascii=False)
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--http_proxy', default='')
    parser.add_argument('--https_proxy', default='')

    args = parser.parse_args()

    main(args)
