
import numpy as np
import pandas as pd
import os

import holidays

from sklearn.preprocessing import LabelEncoder
from dateutil.relativedelta import relativedelta
from datetime import datetime as dt
import datetime
import category_encoders as ce
import random
from sklearn.preprocessing import StandardScaler
import bhtsne

def merge_dataframe(status_df, weather_df, station_df, trip_df, save_df=False):
    """
    main用 全dfを結合 共通の前処理を実行
    [ルール]
    1. 型変換はmerge前に行う
    """

    #  ==================== status_df ====================
    #  status_df の date をdatetime型に変換
    status_df['date'] = status_df['year'].astype(str) + '/' + status_df['month'].astype(str).str.zfill(2).astype(str) + '/' + status_df['day'].astype(str).str.zfill(2).astype(str) + " " + status_df['hour'].astype(str).str.zfill(2).astype(str)
    status_df['date'] = pd.to_datetime(status_df['date'])

    #  ==================== trip_df ====================
    #  mergeするためには集計する必要がある

    #  trip_df の start_date と end_date をdatetime型に変換(高速化のためformat指定)
    trip_df["start_date"] = pd.to_datetime(trip_df["start_date"],format='%m/%d/%Y %H:%M')
    trip_df["end_date"] = pd.to_datetime(trip_df["end_date"],format='%m/%d/%Y %H:%M')
    #  分以下の情報を切り捨てる
    trip_df['start_date'] = trip_df['start_date'].dt.floor("H")
    trip_df['end_date'] = trip_df['end_date'].dt.floor("H")

    #  subscription_typeをone_hot化(この後の集計のため)
    trip_df = pd.get_dummies(trip_df)


    #  station毎に各時間何人の Subscriber がstationから自転車に乗って行ったのかを集計
    trip_hour_start = pd.DataFrame(trip_df.groupby(["start_date","start_station_id"])[["subscription_type_Customer","subscription_type_Subscriber"]].sum()).reset_index()
    trip_hour_start.columns = ["start_date","start_station_id","hour_start_Cus","hour_start_Sub"]
    trip_hour_start["hour_start_Cus"] = trip_hour_start["hour_start_Cus"].astype("uint8")
    trip_hour_start["hour_start_Sub"] = trip_hour_start["hour_start_Sub"].astype("uint8")
    trip_hour_start["hour_start_total"] = trip_hour_start["hour_start_Cus"] + trip_hour_start["hour_start_Sub"]

    trip_hour_end = pd.DataFrame(trip_df.groupby(["end_date","end_station_id"])[["subscription_type_Customer","subscription_type_Subscriber"]].sum()).reset_index()
    trip_hour_end.columns = ["end_date","end_station_id","hour_end_Cus","hour_end_Sub"]
    trip_hour_end["hour_end_Cus"] = trip_hour_end["hour_end_Cus"].astype("uint8")
    trip_hour_end["hour_end_Sub"] = trip_hour_end["hour_end_Sub"].astype("uint8")
    trip_hour_end["hour_end_total"] = trip_hour_end["hour_end_Cus"] + trip_hour_end["hour_end_Sub"]

    #  上記の4変数をmerge
    status_df = pd.merge(status_df, trip_hour_start ,how="left" , left_on=['date', 'station_id'], right_on=['start_date','start_station_id']).drop(columns=['start_date','start_station_id'])
    status_df = pd.merge(status_df, trip_hour_end ,how="left" , left_on=['date','station_id'], right_on=['end_date','end_station_id']).drop(columns=['end_date','end_station_id'])

    del trip_hour_start, trip_hour_end

    #  上で集計した変数は欠損値=0という意味を持つためmergeする前に変換
    trans_col = ["hour_start_Cus","hour_start_Sub","hour_end_Cus","hour_end_Sub"]
    for col in trans_col:
        status_df[col] = status_df[col].fillna(0)
        status_df[col] = status_df[col].astype(int)

    #  status_dfの date の内 時間以下の情報を切り捨てる(この先のmergeのため)
    status_df['date'] = status_df['date'].dt.floor("d")
    del trip_df

    #  ==================== wather_df ====================
    #  weather_df の date をdatetime型に変換
    weather_df["date"] = pd.to_datetime(weather_df['date'])
    weather_df["rainy_day"] = weather_df["precipitation"]>0

    #  status_df に weather_df を merge
    status_df = pd.merge(status_df, weather_df, how = 'left')
    del weather_df

    #  ==================== station_df ====================
    #  station_df の installation_date をdatetime型に変換
    station_df["installation_date"] = pd.to_datetime(station_df['installation_date'])

    train_station_df = pd.read_csv("../data/train_station.csv")

    san_fran_coordinate = [train_station_df[train_station_df["train_station_name"]=="Caltrain Station, San Francisco"]["lat_st"],train_station_df[train_station_df["train_station_name"]=="Caltrain Station, San Francisco"]["long_st"]]
    san_jose_coordinate = [train_station_df[train_station_df["train_station_name"]=="San Jose Dridon station"]["lat_st"],train_station_df[train_station_df["train_station_name"]=="San Jose Dridon station"]["long_st"]]
    for i in range(70):
        min_index = 0
        min_distance = 1000000
        for j in range(train_station_df.shape[0]):
            distance = np.sqrt((station_df["lat"][i]*1000-train_station_df["lat_st"][j]*1000)**2 + (station_df["long"][i]*1000-train_station_df["long_st"][j]*1000)**2)
            if distance<min_distance:
                min_distance = distance
                min_index = j
        station_df.loc[station_df["station_id"]==i,"train_station_id"] = min_index
        station_df.loc[station_df["station_id"]==i,"station_distance"] = min_distance

        #  大都市 San Franciscoまでの距離及び San Joseまでの距離
        """
        to_sanfrancisco = float(np.sqrt((station_df["lat"][i]*1000-san_fran_coordinate[0]*1000)**2 + (station_df["long"][i]*1000-san_fran_coordinate[1]*1000)**2))
        station_df.loc[station_df["station_id"]==i,"to_sanfrancisco"] = to_sanfrancisco
        to_sanjose = float(np.sqrt((station_df["lat"][i]*1000-san_jose_coordinate[0]*1000)**2 + (station_df["long"][i]*1000-san_jose_coordinate[1]*1000)**2))
        station_df.loc[station_df["station_id"]==i,"to_sanjose"] = to_sanjose
        """

    station_df = pd.merge(station_df, train_station_df,how="left",on=["train_station_id"])

    #  status_df に station_df を merge
    status_df = pd.merge(status_df, station_df, how = 'left')
    del station_df

    #  ==================== save ====================
    if save_df:
        status_df.to_csv("../input/merged_df.csv",index=False)
        print("merged_df is saved.")

    return status_df


def common_prosessing(save_df=False, skip_all_flag=True, skip_merge_flag=True):
    """
    全てのデータフレームがmergeされたデータフレームから新たな特徴量を生成
    """

    skip_all_flag = skip_all_flag# all_dfがあっても書き換えたい時はFalseへ	
    path = "../input/all_df.csv"
    if(skip_merge_flag and skip_all_flag and os.path.exists(path)):
        print("all_df is exists.")
        all_df = pd.read_csv(path)
        col = ["date","installation_date"]
        for c in col:
            all_df[c] = pd.to_datetime(all_df[c])
        return all_df


    #  単にmergeされたfileを作成 (既にmerge済みのファイルがあればそれを返す)
    skip_merge_flag = skip_merge_flag# merge_dfから書き換えたい時はFalseへ	
    path = "../input/merged_df.csv"
    if (skip_merge_flag and os.path.exists(path)):
        print("merged_df is exists.")
        all_df = pd.read_csv(path)
        col = ["date","installation_date"]
        for c in col:
            all_df[c] = pd.to_datetime(all_df[c])

    else:
        status_df = pd.read_csv("../data/status.csv")
        weather_df = pd.read_csv("../data/weather.csv")
        station_df = pd.read_csv("../data/station.csv")
        trip_df = pd.read_csv("../data/trip.csv")
        all_df = merge_dataframe(status_df, weather_df, station_df, trip_df, save_df=True)
        del status_df, weather_df, station_df, trip_df


    # ========== 関数main処理開始 ====================

    #  変数組み替え用スイッチ(見やすい)
    week_num = True
    holiday = True
    yr_month = True
    bikes_available_at0 = True
    from_installation_date = True
    temperature_difference = True
    season = True
    dock_tile = False
    external_at0 = False
    ratio_at0 = True
    fluctuation = True

    #  ==================== merged_dfから生成 ====================
    #  曜日の変数を特徴量に追加
    if week_num:
        all_df['week_num'] = all_df['date'].dt.weekday

    #  休日か否かの特徴量
    if holiday:

        us_holidays = holidays.US()
        all_df['holiday'] = all_df['date'].map(lambda d:1 if d in us_holidays else 0)

        all_df.loc[all_df["week_num"]>=5,"holiday"] = 1
        all_df['holiday_1day_lag'] = all_df['holiday'].fillna(0)

    #  月の通し番号を特徴量に追加
    if yr_month:
        all_df['yr_month'] = all_df["year"].astype(str)+"_"+all_df["month"].astype(str)
        for i,s in enumerate(all_df["yr_month"].unique()):
            all_df.loc[all_df["yr_month"]==s,"yr_month"] = i
        all_df["yr_month"] = all_df["yr_month"].astype(int)

    #  同じステーション、同じ日において、0時時点の台数を特徴量に追加
    if bikes_available_at0:
        t = all_df.groupby(['station_id', 'date']).first()['bikes_available'].reset_index()
        t = pd.DataFrame(np.repeat(t.values, 24, axis=0))
        t.columns = ['station_id', 'date', 'bikes_available_at0']
        all_df['bikes_available_at0'] = t['bikes_available_at0']
        all_df['bikes_available_at0'] = all_df['bikes_available_at0'].astype(float)

    #  設置から経った日数
    if from_installation_date:
        all_df["from_installation_date"] = (all_df['date']-all_df['installation_date']).map(lambda d: d.days)

    #  気温差
    if temperature_difference:
         all_df["temperature_difference"] = all_df["max_temperature"] - all_df["min_temperature"]

    #  季節
    if season:
        season_map = {1:4,2:1,3:1,4:1,5:2,6:2,7:2,8:3,9:3,10:3,11:4,12:4}
        all_df['season'] = all_df["month"].map(season_map)


    #  object型の変数をLabel Encoding
    for c in ["events","city","Route"]:
        encoder = LabelEncoder()
        temp_ = pd.DataFrame(encoder.fit_transform(all_df[c]),columns=[c]).add_prefix("LE_")
        all_df = pd.concat([all_df, temp_], axis=1)
        all_df.drop(c, axis=1)

    #  ==================== 上記の変数から生成 ====================

    if ratio_at0:
        #  上記の変数の拡張版
        all_df["ratio_at0"] = all_df["bikes_available_at0"]/all_df["dock_count"]

    if fluctuation:
        #  値の変動
        all_df.loc[~all_df["bikes_available"].isnull(),"fluctuation"] = all_df["bikes_available"] - all_df["bikes_available_at0"]

    #  ==================== 目的変数欠損値埋め >> lag変数作成 ====================

    all_df.loc[(all_df["bikes_available"].isnull())&(all_df["predict"]==0),"predict"] = -1

    all_df_new = pd.DataFrame()
    for sid in range(70):
        temp_df = all_df[all_df["station_id"]==sid].copy()
        temp_df.loc[:,"bikes_available"] = temp_df["bikes_available"].fillna(method="ffill")

        #  1h~3hlag変数 24~26hlag変数
        for n_shift in range(3):
            col_name = "bikes_available_"+str(n_shift+1)+"hour_lag"
            temp_df.loc[:,col_name] = temp_df["bikes_available"].shift(n_shift+1).fillna(method="bfill")

            col_name = "bikes_available_"+str(n_shift+23)+"hour_lag"
            temp_df.loc[:,col_name] = temp_df["bikes_available"].shift(n_shift+23).fillna(method="bfill")

        all_df_new = pd.concat([all_df_new,temp_df])

    all_df = all_df_new
    del temp_df, all_df_new

    #  ==================== save ====================
    if save_df:
        all_df.to_csv("../input/all_df.csv",index=False)
        print("all_df is saved.")

    return all_df


def time_processing(all_df, trip_df, start_period=None, end_period=None):
    """
    リークを考慮した前処理
    引数 : 共通前処理が完了した全てのデータフレーム
    戻り値 : 時系列に関わる処理をし終わったデータフレーム
    毎回作るためセーブ機能は無し

    train用と予測用で特徴量作り直す必要あり
    もしかしたらtrainの期間狭めの方が良い可能性もあり, 検証必要
    もしかしたら曜日毎とかの集計もあるかも

    object型を変換
    """
    print(" ========================================================= ")
    print("[ time_processing start. ]")
    print("time_processing refer ",start_period," >> ",end_period,".")

    #  ================ (重要)trainデータのみの領域  ================

    start_period = dt.strptime(start_period, '%Y-%m-%d')
    end_period = dt.strptime(end_period, '%Y-%m-%d')

    #  範囲切り出し
    section_trip_df = trip_df[trip_df["end_date"]<end_period]

    #  A.各ステーション, 時間毎のsubscription_type関連
    #  week hour単位
    td = datetime.timedelta(weeks=26)# 6ヶ月分
    trip_wh_start = pd.DataFrame(section_trip_df[end_period-td <= section_trip_df["start_date"]].groupby(["start_station_id","start_week","start_hour"])[["subscription_type_Customer","subscription_type_Subscriber"]].sum()).reset_index()
    trip_wh_start.columns = ["start_station_id","start_week","start_hour","wh_start_Cus","wh_start_Sub"]
    trip_wh_start["wh_start_Cus"] = trip_wh_start["wh_start_Cus"].astype("uint8")
    trip_wh_start["wh_start_Sub"] = trip_wh_start["wh_start_Sub"].astype("uint8")
    trip_wh_start["wh_start_total"] = trip_wh_start["wh_start_Cus"] + trip_wh_start["wh_start_Sub"]
    trip_wh_start["wh_start_Sub_ratio"] = trip_wh_start["wh_start_Sub"]/trip_wh_start["wh_start_total"]

    trip_wh_end = pd.DataFrame(section_trip_df[end_period-td <= section_trip_df["start_date"]].groupby(["end_station_id","end_week","end_hour"])[["subscription_type_Customer","subscription_type_Subscriber"]].sum()).reset_index()
    trip_wh_end.columns = ["end_station_id","end_week","end_hour","wh_end_Cus","wh_end_Sub"]
    trip_wh_end["wh_end_Cus"] = trip_wh_end["wh_end_Cus"].astype("uint8")
    trip_wh_end["wh_end_Sub"] = trip_wh_end["wh_end_Sub"].astype("uint8")
    trip_wh_end["wh_end_total"] = trip_wh_end["wh_end_Cus"] + trip_wh_end["wh_end_Sub"]
    trip_wh_end["wh_end_Sub_ratio"] = trip_wh_end["wh_end_Sub"]/trip_wh_end["wh_end_total"]

    #  testデータを含む全てのデータにmerge
    time_df = pd.merge(all_df, trip_wh_start, how="left" , left_on=["station_id","week_num","hour"], right_on=["start_station_id","start_week","start_hour"]).drop(columns=["start_station_id","start_week","start_hour"])
    time_df = pd.merge(time_df, trip_wh_end, how="left" , left_on=["station_id","week_num","hour"], right_on=["end_station_id","end_week","end_hour"]).drop(columns=["end_station_id","end_week","end_hour"])

    time_df["wh_start_Sub_ratio"] = time_df["wh_start_Sub_ratio"].fillna(0.5)
    time_df["wh_end_Sub_ratio"] = time_df["wh_end_Sub_ratio"].fillna(0.5)
    time_df["wh_trip_fluc"] = time_df["wh_end_total"] - time_df["wh_start_total"]

    #  引き継ぎ防止
    del trip_wh_start, trip_wh_end

    #  単純曜日単位
    td = datetime.timedelta(weeks=13)# 3ヶ月分
    trip_week_start = pd.DataFrame(section_trip_df[end_period-td <= section_trip_df["start_date"]].groupby(["start_station_id","start_week"])[["subscription_type_Customer","subscription_type_Subscriber"]].sum()).reset_index()
    trip_week_start.columns = ["start_station_id","start_week","week_start_Cus","week_start_Sub"]
    trip_week_start["week_start_Cus"] = trip_week_start["week_start_Cus"]
    trip_week_start["week_start_Sub"] = trip_week_start["week_start_Sub"]
    trip_week_start["week_start_total"] = trip_week_start["week_start_Cus"] + trip_week_start["week_start_Sub"]
    trip_week_start["week_start_Sub_ratio"] = trip_week_start["week_start_Sub"]/trip_week_start["week_start_total"]
    trip_week_start = trip_week_start.fillna(0.5)

    trip_week_end = pd.DataFrame(section_trip_df[end_period-td <= section_trip_df["start_date"]].groupby(["end_station_id","end_week"])[["subscription_type_Customer","subscription_type_Subscriber"]].sum()).reset_index()
    trip_week_end.columns = ["end_station_id","end_week","week_end_Cus","week_end_Sub"]
    trip_week_end["week_end_Cus"] = trip_week_end["week_end_Cus"]
    trip_week_end["week_end_Sub"] = trip_week_end["week_end_Sub"]
    trip_week_end["week_end_total"] = trip_week_end["week_end_Cus"] + trip_week_end["week_end_Sub"]
    trip_week_end["week_end_Sub_ratio"] = trip_week_end["week_end_Sub"]/trip_week_end["week_end_total"]
    trip_week_end = trip_week_end.fillna(0.5)

    #  testデータを含む全てのデータにmerge
    time_df = pd.merge(time_df, trip_week_start, how="left" , left_on=["station_id","week_num"], right_on=["start_station_id","start_week"]).drop(columns=["start_station_id","start_week"])
    time_df = pd.merge(time_df, trip_week_end, how="left" , left_on=["station_id","week_num"], right_on=["end_station_id","end_week"]).drop(columns=["end_station_id","end_week"])

    time_df["week_start_Sub_ratio"] = time_df["week_start_Sub_ratio"].fillna(0.5)
    time_df["week_end_Sub_ratio"] = time_df["week_end_Sub_ratio"].fillna(0.5)
    time_df["week_trip_fluc"] = time_df["week_end_total"] - time_df["week_start_total"]

    del trip_week_start, trip_week_end


    #  station_id単位
    td = datetime.timedelta(weeks=13)# 3ヶ月分
    trip_sid_start = pd.DataFrame(section_trip_df[end_period-td <= section_trip_df["start_date"]].groupby(["start_station_id"])[["subscription_type_Customer","subscription_type_Subscriber"]].sum()).reset_index()
    trip_sid_start.columns = ["start_station_id","sid_start_Cus","sid_start_Sub"]
    trip_sid_start["sid_start_Cus"] = trip_sid_start["sid_start_Cus"]
    trip_sid_start["sid_start_Sub"] = trip_sid_start["sid_start_Sub"]
    trip_sid_start["sid_start_total"] = trip_sid_start["sid_start_Cus"] + trip_sid_start["sid_start_Sub"]
    trip_sid_start["sid_start_Sub_ratio"] = trip_sid_start["sid_start_Sub"]/trip_sid_start["sid_start_total"]
    trip_sid_start = trip_sid_start.fillna(0.5)

    trip_sid_end = pd.DataFrame(section_trip_df[end_period-td <= section_trip_df["start_date"]].groupby(["end_station_id"])[["subscription_type_Customer","subscription_type_Subscriber"]].sum()).reset_index()
    trip_sid_end.columns = ["end_station_id","sid_end_Cus","sid_end_Sub"]
    trip_sid_end["sid_end_Cus"] = trip_sid_end["sid_end_Cus"]
    trip_sid_end["sid_end_Sub"] = trip_sid_end["sid_end_Sub"]
    trip_sid_end["sid_end_total"] = trip_sid_end["sid_end_Cus"] + trip_sid_end["sid_end_Sub"]
    trip_sid_end["sid_end_Sub_ratio"] = trip_sid_end["sid_end_Sub"]/trip_sid_end["sid_end_total"]
    trip_sid_end = trip_sid_end.fillna(0.5)

    #  testデータを含む全てのデータにmerge
    time_df = pd.merge(time_df, trip_sid_start, how="left" , left_on=["station_id"], right_on=["start_station_id"]).drop(columns=["start_station_id"])
    time_df = pd.merge(time_df, trip_sid_end, how="left" , left_on=["station_id"], right_on=["end_station_id"]).drop(columns=["end_station_id"])

    time_df["sid_start_Sub_ratio"] = time_df["sid_start_Sub_ratio"].fillna(0.5)
    time_df["sid_end_Sub_ratio"] = time_df["sid_end_Sub_ratio"].fillna(0.5)
    time_df["sid_trip_fluc"] = time_df["sid_end_total"] - time_df["sid_start_total"]

    del trip_sid_start, trip_sid_end

    cols = ['wh_start_Cus', 'wh_start_Sub', 'wh_start_total',
        'wh_end_Cus', 'wh_end_Sub', 'wh_end_total',
        'wh_trip_fluc', 'week_start_Cus', 'week_start_Sub', 'week_start_total',
        'week_end_Cus', 'week_end_Sub',
        'week_end_total', 'week_trip_fluc',
        'sid_start_Cus', 'sid_start_Sub', 'sid_start_total',
        'sid_end_Cus', 'sid_end_Sub', 'sid_end_total', 'sid_trip_fluc']
    time_df[cols] = time_df[cols].fillna(0)

    del section_trip_df

    #  trainデータの区間で区切ったDataFrameの作成, リークしないようにこれを計算に使う
    section_df = all_df[(start_period<=all_df["date"])&(all_df["date"]<end_period)]
    section_df = section_df[0<=section_df["from_installation_date"]]

    #  ============================== 以降 目的変数をもとにした前処理 ===========================

    #  === section_df作り直し ===
    section_df = section_df[~section_df["bikes_available"].isnull()]
    #  補正した部分は除く
    section_df = section_df[section_df["predict"]!=-1]


    #  B.目的変数も用いた変換(範囲内でfluctuation再計算)
    section_df["fluctuation"] = section_df["bikes_available"]-section_df["bikes_available_at0"]
    td = datetime.timedelta(weeks=13)# 3ヶ月分
    temp_df = pd.DataFrame(section_df[(end_period-td)<=section_df["date"]].groupby(["station_id","week_num","hour"])["fluctuation"].mean()).reset_index()
    temp_df.columns = ["station_id","week_num","hour","3mean_sid_fluctuation"]

    time_df = pd.merge(time_df, temp_df, how="left", on=["station_id","week_num","hour"])

    td = datetime.timedelta(weeks=13)# 3ヶ月分
    temp_df = pd.DataFrame(section_df[(end_period-td)<=section_df["date"]].groupby(["city","week_num","hour"])["fluctuation"].mean()).reset_index()
    temp_df.columns = ["city","week_num","hour","3mean_city_fluctuation"]

    time_df = pd.merge(time_df, temp_df, how="left", on=["city","week_num","hour"])

    td = datetime.timedelta(weeks=13)# 3ヶ月分
    temp_df = pd.DataFrame(section_df[(end_period-td)<=section_df["date"]].groupby(["station_id","week_num","hour"])["bikes_available"].mean()).reset_index()
    temp_df.columns = ["station_id","week_num","hour","3mean_sid_fluctuation"]

    time_df = pd.merge(time_df, temp_df, how="left", on=["station_id","week_num","hour"])

    td = datetime.timedelta(weeks=13)# 3ヶ月分
    temp_df = pd.DataFrame(section_df[(end_period-td)<=section_df["date"]].groupby(["city","week_num","hour"])["bikes_available"].mean()).reset_index()
    temp_df.columns = ["city","week_num","hour","3mean_city_fluctuation"]

    time_df = pd.merge(time_df, temp_df, how="left", on=["city","week_num","hour"])

    #  引き継ぎ防止
    del temp_df

    #  C.target encoding
    #  B同様,bikes_availableが欠損値でないかつ補正した値ではない場所で計算(fluctuationを元に)
    td = datetime.timedelta(weeks=13)# 3ヶ月分
    te = ce.LeaveOneOutEncoder(cols=['station_id']) 
    #  fitは特定の範囲だけ
    te.fit(section_df[(end_period-td)<=section_df["date"]]['station_id'], section_df[(end_period-td)<=section_df["date"]]['bikes_available'])
    #  transformは全て
    time_df['target_enc_sid'] = te.transform(time_df['station_id'])

    td = datetime.timedelta(weeks=13)# 3ヶ月分
    te = ce.LeaveOneOutEncoder(cols=['city']) 
    #  fitは特定の範囲だけ
    te.fit(section_df[(end_period-td)<=section_df["date"]]['city'], section_df[(end_period-td)<=section_df["date"]]['bikes_available'])
    #  transformは全て
    time_df['target_enc_city'] = te.transform(time_df['city'])

    #  B同様,bikes_availableが欠損値でないかつ補正した値ではない場所で計算(fluctuationを元に)
    td = datetime.timedelta(weeks=13)# 3ヶ月分
    te = ce.LeaveOneOutEncoder(cols=['station_id']) 
    #  fitは特定の範囲だけ
    te.fit(section_df[(end_period-td)<=section_df["date"]]['station_id'], section_df[(end_period-td)<=section_df["date"]]['fluctuation'])
    #  transformは全て
    time_df['fluc_enc_sid'] = te.transform(time_df['station_id'])

    td = datetime.timedelta(weeks=13)# 3ヶ月分
    te = ce.LeaveOneOutEncoder(cols=['city']) 
    #  fitは特定の範囲だけ
    te.fit(section_df[(end_period-td)<=section_df["date"]]['city'], section_df[(end_period-td)<=section_df["date"]]['fluctuation'])
    #  transformは全て
    time_df['fluc_enc_city'] = te.transform(time_df['city'])

    #  D.ランダムでtrainデータの中のラグ変数をいじる
    random_lag = False
    if random_lag:
        random.seed(0)
        for n_shift in range(3):
            col_name = "bikes_available_"+str(n_shift+1)+"hour_lag"
            f = lambda col: col["bikes_available_at0"] if random.random()<0.2 else col[col_name]
            time_df.loc[(start_period<=time_df["date"])&(time_df["date"]<end_period),col_name] = time_df[(start_period<=time_df["date"])&(time_df["date"]<end_period)].apply(f, axis='columns')

    #  E.full-empty特徴量
    td = datetime.timedelta(weeks=13)# 3ヶ月分
    temp_df = section_df[(end_period-td)<=section_df["date"]]
    temp_df.loc[:,"sid_empty_count"] = (temp_df["bikes_available"]<3)
    temp_df.loc[:,"sid_full_count"] = (temp_df["bikes_available"]>(temp_df["dock_count"]-3))
    temp_df = pd.DataFrame(temp_df.groupby(["station_id","week_num","hour"])[["sid_empty_count","sid_full_count"]].sum()).reset_index()
    temp_df.columns = ["station_id","week_num","hour","sid_empty_count","sid_full_count"]

    time_df = pd.merge(time_df, temp_df, how="left", on=["station_id","week_num","hour"])

    del temp_df

    td = datetime.timedelta(weeks=13)# 3ヶ月分
    temp_df = section_df[(end_period-td)<=section_df["date"]]
    temp_df.loc[:,"city_empty_count"] = (temp_df["bikes_available"]<3)
    temp_df.loc[:,"city_full_count"] = (temp_df["bikes_available"]>(temp_df["dock_count"]-3))
    temp_df = pd.DataFrame(temp_df.groupby(["city","week_num","hour"])[["city_empty_count","city_full_count"]].sum()).reset_index()
    temp_df.columns = ["city","week_num","hour","city_empty_count","city_full_count"]

    time_df = pd.merge(time_df, temp_df, how="left", on=["city","week_num","hour"])

    del temp_df

    #  F.変動具合分散特徴量
    td = datetime.timedelta(weeks=13)# 3ヶ月分
    temp_df = section_df[(end_period-td)<=section_df["date"]]
    temp_df = pd.DataFrame(temp_df.groupby(["station_id","week_num","hour"])["bikes_available"].var()).reset_index()
    temp_df.columns = ["station_id","week_num","hour","var_sid_bikes_available"]

    time_df = pd.merge(time_df, temp_df, how="left", on=["station_id","week_num","hour"])

    del temp_df

    td = datetime.timedelta(weeks=13)# 3ヶ月分
    temp_df = section_df[(end_period-td)<=section_df["date"]]
    temp_df = pd.DataFrame(temp_df.groupby(["station_id","week_num","hour"])["bikes_available"].var()).reset_index()
    temp_df.columns = ["station_id","week_num","hour","var_city_bikes_available"]

    time_df = pd.merge(time_df, temp_df, how="left", on=["station_id","week_num","hour"])

    del temp_df


    """
    #  ============================== t-SNE ===========================
    #  trainデータの区間で区切ったDataFrameの作成, リークしないようにこれを計算に使う
    section_df = all_df[(start_period<=all_df["date"])&(all_df["date"]<end_period)]
    section_df = section_df[0<=section_df["from_installation_date"]]
    use_cols = ['month', 'day', 'hour', 'station_id','target_enc_sid','target_enc_city','fluc_enc_sid','fluc_enc_city',
        'max_temperature', 'mean_temperature', 'min_temperature', 'temperature_difference',
        'mean_dew_point','mean_humidity','mean_sea_level_pressure','max_wind_Speed', 'mean_wind_speed', 'precipitation',
        'lat', 'long', 'dock_count', 'from_installation_date',
        'week_num', 'holiday','yr_month','ratio_at0',
        'wh_start_Cus', 'wh_start_Sub', 'wh_start_total', 'wh_start_Sub_ratio',
        'wh_end_Cus', 'wh_end_Sub', 'wh_end_total', 'wh_end_Sub_ratio',
        'wh_trip_fluc', 'week_start_Cus', 'week_start_Sub', 'week_start_total',
        'week_start_Sub_ratio', 'week_end_Cus', 'week_end_Sub',
        'week_end_total', 'week_end_Sub_ratio', 'week_trip_fluc',
        'bikes_available_23hour_lag', 'bikes_available_24hour_lag','bikes_available_25hour_lag',
        "sid_empty_count", "sid_full_count", "city_empty_count", "city_full_count","var_sid_bikes_available","var_city_bikes_available"]
    temp_df = section_df[use_cols].dropna(how='any', axis=1)
    sc = StandardScaler()
    print(temp_df.info())
    temp_df  = pd.DataFrame(sc.fit_transform(temp_df),columns=temp_df.columns)
    embedded = bhtsne.tsne(temp_df.astype(np.float64),dimensions=2,rand_seed=71)
    """
    

    print("time_processing end.")
    print(" ========================================================= \n")

    return time_df

def data_split(time_df, start_train=None, end_train=None, start_valid=None, end_valid=None, start_test=None, end_test=None, test_mode=False):

    if test_mode:

        #  test分割
        test_data = time_df[time_df["predict"]==1]

        #  testデータ準備
        test_X = test_data.drop(["bikes_available"],axis=1)

        #  testデータから予測対象期間を抽出
        X_test = test_X[(start_test<=test_X["date"])&(test_X["date"]<end_test)]

        return X_test

    else:

        #  train分割
        train_data = time_df[time_df["predict"]==0]
        train_data = train_data[train_data['bikes_available'].notna()]
        train_data = train_data[train_data['hour']!=0]# 0時の列も落とす
        
        #  学習データ準備
        train_X = train_data.drop(["bikes_available"],axis=1)
        train_Y = train_data["bikes_available"]

        #  trainデータをvalidation用に分割
        X_train = train_X[(start_train<=train_X["date"])&(train_X["date"]<end_train)]
        y_train = train_Y[(start_train<=train_X["date"])&(train_X["date"]<end_train)]
        X_valid = train_X[(start_valid<=train_X["date"])&(train_X["date"]<end_valid)]
        y_valid = train_Y[(start_valid<=train_X["date"])&(train_X["date"]<end_valid)]
        X_train_test = train_X[(start_test<=train_X["date"])&(train_X["date"]<end_test)]
        y_train_test = train_Y[(start_test<=train_X["date"])&(train_X["date"]<end_test)]

        return X_train, y_train, X_valid, y_valid, X_train_test, y_train_test


if __name__ == "__main__":
    
    all_df = common_prosessing(save_df=True, skip_all_flag=False, skip_merge_flag=True)# all_dfの保存を修正したい場合
    all_df.info()
    print(all_df[["date","hour","station_id","bikes_available_23hour_lag","bikes_available_24hour_lag","bikes_available_25hour_lag"]].head(30))

    #  時系列前処理test → test成功
    # time_df = time_processing(all_df, start_period="2013-09-01", end_period="2014-09-01")

