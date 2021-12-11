

def main():
    pass

if __name__=='__main__':
    import pandas as pd

    horse_id = "2018103295"

    url = 'https://db.netkeiba.com/horse/' + horse_id

    df_horse = pd.read_html(url, match='レース名')[0]

    df_horse.to_csv('./data/horse_data/' + horse_id + '.csv', index=False)