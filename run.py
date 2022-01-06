# -*- coding: utf-8 -*-

from pre_processing import *
from models import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
import datetime

from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import optuna

pd.set_option('display.max_rows', 100)

def train_and_predict(models, all_df):

    #  ========== パラメータ ==========

    #  学習に使う列
    use_cols = ['month', 'day', 'hour', 'station_id','target_enc_sid','target_enc_city','fluc_enc_sid','fluc_enc_city',

        'max_temperature', 'mean_temperature', 'min_temperature', 'temperature_difference',
        'mean_dew_point','mean_humidity','mean_sea_level_pressure','max_wind_Speed', 'mean_wind_speed', 'precipitation',

        'lat', 'long', 'dock_count', 'from_installation_date',

        'week_num', 'holiday','holiday_1day_lag','yr_month','ratio_at0',

        'wh_start_Cus', 'wh_start_Sub', 'wh_start_total', 'wh_start_Sub_ratio',
        'wh_end_Cus', 'wh_end_Sub', 'wh_end_total', 'wh_end_Sub_ratio',
        'wh_trip_fluc', 'week_start_Cus', 'week_start_Sub', 'week_start_total',
        'week_start_Sub_ratio', 'week_end_Cus', 'week_end_Sub',
        'week_end_total', 'week_end_Sub_ratio', 'week_trip_fluc',
        'sid_start_Cus', 'sid_start_Sub', 'sid_start_total',
        'sid_start_Sub_ratio', 'sid_end_Cus', 'sid_end_Sub', 'sid_end_total',
        'sid_end_Sub_ratio', 'sid_trip_fluc',

        'station_distance',

        'bikes_available_23hour_lag', 'bikes_available_24hour_lag','bikes_available_25hour_lag',

        "sid_empty_count", "sid_full_count", "city_empty_count", "city_full_count","var_sid_bikes_available","var_city_bikes_available"]


    for model in models:
        model.use_cols = use_cols

    #  カテゴリカルデータ指定
    categories = ['station_id','holiday']

    #  trainデータの範囲指定
    #  starts_of_train = ["2013-09-01","2013-09-01","2013-09-01","2013-09-01","2013-09-01","2013-09-01","2013-09-01","2013-09-01","2013-09-01","2013-09-01","2013-09-01","2013-09-01"]
    starts_of_train = ["2013-09-01","2013-10-01","2013-11-01","2013-12-01","2014-01-01","2014-02-01","2014-03-01","2014-04-01","2014-05-01","2014-06-01","2014-07-01","2014-08-01"]
    #  starts_of_train = ["2014-05-01","2014-06-01","2014-07-01","2014-08-01","2014-09-01","2014-10-01","2014-11-01","2014-12-01","2015-01-01","2015-02-01","2015-03-01","2015-04-01",]
    ends_of_train = ["2014-08-01","2014-09-01","2014-10-01","2014-11-01","2014-12-01","2015-01-01","2015-02-01","2015-03-01","2015-04-01","2015-05-01","2015-06-01","2015-07-01"]

    #  validデータの範囲指定
    starts_of_valid = ["2014-08-01","2014-09-01","2014-10-01","2014-11-01","2014-12-01","2015-01-01","2015-02-01","2015-03-01","2015-04-01","2015-05-01","2015-06-01","2015-07-01"]
    ends_of_valid = ["2014-09-01","2014-10-01","2014-11-01","2014-12-01","2015-01-01","2015-02-01","2015-03-01","2015-04-01","2015-05-01","2015-06-01","2015-07-01","2015-08-01"]
        
    #  testデータの範囲指定
    starts_of_test = ["2014-09-01","2014-10-01","2014-11-01","2014-12-01","2015-01-01","2015-02-01","2015-03-01","2015-04-01","2015-05-01","2015-06-01","2015-07-01","2015-08-01"]
    ends_of_test = ["2014-10-01","2014-11-01","2014-12-01","2015-01-01","2015-02-01","2015-03-01","2015-04-01","2015-05-01","2015-06-01","2015-07-01","2015-08-01","2015-09-01"]


    #  ========== データの箱用意 ==========

    #  データの並び変更
    all_df = all_df[all_df["date"]<"2015-09-01"]
    all_df = all_df.sort_values(["date","hour","station_id"],ascending=True).reset_index(drop=True)
    print(all_df.info())
    print(all_df.head())
    print(all_df.tail())

    #  提出用ファイル
    sample_df = pd.read_csv("../data/sample_submit.csv",header=None)
    #  print(sample_df.shape) 70station 12month 10day 23hour = 193200行

    #  予測値保存用
    preds_array = np.zeros((len(models), sample_df.shape[0]), dtype=float)
    at0_array = np.zeros((len(models), sample_df.shape[0]), dtype=float)

    #  ========== 関数のmain処理 ========== 
    for i, (st, et, sv, ev, stes, etes) in enumerate(zip(starts_of_train, ends_of_train, starts_of_valid, ends_of_valid, starts_of_test, ends_of_test)):

        print(" =========================== Each Period ============================= ")
        print("train_data: ", st, " >> ", et)
        print("valid_data: ", sv, " >> ", ev)
        print("test_data: ", stes, " >> ", etes)
        print(" ===================================================================== \n")

        #  1.学習用時系列前処理
        if i==0:
            trip_df = pd.read_csv("../data/trip.csv")
            #  trip_df前処理
            trip_df["start_date"] = pd.to_datetime(trip_df["start_date"],format='%m/%d/%Y %H:%M')
            trip_df["end_date"] = pd.to_datetime(trip_df["end_date"],format='%m/%d/%Y %H:%M')
            trip_df['start_date'] = trip_df['start_date'].dt.floor("H")
            trip_df['end_date'] = trip_df['end_date'].dt.floor("H")
            trip_df['start_week'] = trip_df['start_date'].dt.weekday
            trip_df['end_week'] = trip_df['end_date'].dt.weekday
            trip_df['start_hour'] = trip_df['start_date'].map(lambda d: d.hour)
            trip_df['end_hour'] = trip_df['end_date'].map(lambda d: d.hour)
            trip_df = pd.get_dummies(trip_df)

            time_df = time_processing(all_df, trip_df, start_period=st, end_period=et)#  stが同じ場合 evが次のetになるためこの計算は最初だけで良い
            print("columns: ",time_df.columns)
            print(time_df[use_cols].info())

        #  2.学習用にデータを分割
        X_train, y_train, X_valid, y_valid, X_train_test, y_train_test = data_split(time_df, start_train=st, end_train=et, start_valid=sv, end_valid=ev, start_test=stes, end_test=etes)

        for j,model in enumerate(models):
            #  3.学習
            print(" ============= Train Start >> [ class name = ",model.__class__.__name__," ] ============ ")

            model.train(X_train, y_train, X_valid, y_valid, X_train_test, y_train_test, categories=categories, show_graph=True, print_importance=True, show_importance=False)

            print(" ===================================== Train End ======================================= \n")

        #  1.test用時系列前処理(validationの終わりまでを使用)
        time_df = time_processing(all_df, trip_df, start_period=st, end_period=ev)

        #  2.予測用にデータを分割
        X_test = data_split(time_df, trip_df, start_test=stes, end_test=etes, test_mode=True) 
        #  print(X_test.shape) 70station 1month 10day 23hour = 16100行
        #  address: 16100*12 = 193200行分

        for j, model in enumerate(models):
            #  3.予測
            print(" ============= Predict Start >> [ class name = ",model.__class__.__name__," ] ============ ")

            X_test["preds"] = model.predict(X_test)
            X_test = X_test.sort_values(["station_id","date","hour"],ascending=True).reset_index(drop=True)

            for sid in range(70):

                start_preds = sid*2760 + i*230
                end_preds = start_preds + 230
                preds_array[j][start_preds:end_preds] = np.array(X_test["preds"][230*sid:230*sid+230])
                at0_array[j][start_preds:end_preds] = np.array(X_test["bikes_available_at0"][230*sid:230*sid+230])

            print(" ===================================== Predict End ======================================= \n")

    for k, model in enumerate(models):

        print(" ============= Create Submit File >> [ class name = ",model.__class__.__name__," ] ============ ")
        score = np.mean(model.train_test_scores)
        print("train_test_score: ",score)
        score = int(round(score,4)*10000)

        pred = preds_array[k]
        at0 = at0_array[k]
        if(len(sample_df[1])==len(pred)):

            file_name = "../submit/"+model.__class__.__name__+"_1022_lr"+str(int(model.learning_rate*100))+"_"+str(model.random_seed_num)+"seed"+"_"+str(score)+".csv"
            sample_df[1]=pd.DataFrame(pred)
            sample_df.to_csv(file_name,index=False, header=False)
            print(sample_df.info())
            print("file_name: [ ",file_name," ]")

        else:
            print("error: 長さが異なります")

        print(" ===================================== Save is done ======================================= \n")


if __name__ == '__main__':

    #  モデルの定義, 初期化
    lgb_model = Lgb_model()
    rdf_model = Rdf_model()
    models = [lgb_model]

    #  全データ共通の前処理後のデータ 取得・表示
    all_df = common_prosessing(save_df=True)

    #  学習, 予測の開始
    train_and_predict(models, all_df)

    