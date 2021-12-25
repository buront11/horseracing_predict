import sqlite3

import pandas as pd

# TODO 後でやる

class HorseDB():
    def __init__(self, recreate_table=False):
        db_name = 'horse_racing.db'
        self.conn = sqlite3.connect(db_name, isolation_level=None)

        self.cursor = self.conn.cursor()

        if recreate_table:
            self.del_table('race')
            self.del_table('horse')

        sql = '''create table if not exists 
        race(race_id, title, sub_title, round, date, year, month, day, \
        location, race_number, arrival, frame_number, horse_number, horse_name,\
        sex_old, weight, jockey, time, arrival_diff, win_rate,\
        popularity, horse_weight, trainer, race_type, condition,\
        distance, weather, race_line, rotation, start_time)'''

        self.cursor.execute(sql)

        sql = '''create table if not exists
        horse(horse_id, date, horse_name, passing, pace, 3f, winning)
        '''

        self.cursor.execute(sql)
        self.conn.commit()
        
        self.race_id = 1
        self.horse_id = 1

    def _race_col_order_convertion(self, path):
        race_df = pd.read_csv(path)
        race_df['race_id'] = self.race_id
        df = race_df.loc[:,["race_id","title","sub_title","round","date","year","month","day",
        "location","race_num","着順","枠番","馬番","馬名","性齢","斤量","騎手","タイム","着差","単勝",
        "人気","馬体重","調教師","race_type","condition","distance","weather","race_line","rotation","start_time"]]

        return df

    def _horse_col_order_convertion(self, horse_df):
        horse_df['horse_id'] = self.horse_id
        df = horse_df.loc[:,["horse_id", "日付", "馬名", "通過"]]

    def insert_data(self, df, table):
        if table == 'race':
            sql = """INSERT INTO race VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
            self.race_id += 1
            df = self._race_col_order_convertion(df)
        elif table == 'horse':
            sql = """INSERT INTO test VALUES(?,?,?,?,?,?,?)"""
            self.horse_id += 1
            df = self._horse_col_order_convertion(df)

        datas = list(df.itertuples(index=False, name=None))

        self.cursor.executemany(sql, datas)
        self.conn.commit()

    def del_table(self, table):
        sql = f"""DROP TABLE if exists {table}"""

        self.conn.execute(sql)
        self.conn.commit()

    def feach_table(self, table):
        sql = f"""SELECT * FROM {table}"""
        self.cursor.execute(sql)
        print(self.cursor.fetchall())#全レコードを取り出す

if __name__=='__main__':
    db = HorseDB()

    df = pd.read_csv('./data/race_data/200801010101.csv')

    db.insert_data(df, 'race')

    db.feach_table('race')

