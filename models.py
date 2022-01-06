# -*- coding: utf-8 -*-

import pre_processing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

import optuna


class Lgb_model:
    def __init__(self):

        # ========== パラメータ ==========
        self.train_counter = 0# 後のpath名用
        self.random_seed_num = 1
        self.num_boost_round = 10000# 10000
        self.learning_rate = 0.005
        self.params = {

            # 固定パラメータ
            "objective":"regression",
            "metric":"rmse",
            "learning_rate":self.learning_rate,
            "random_seed":1,
            "force_col_wise":True,

            # 可変パラメータ
            'max_depth': 11,
            'num_leaves': 23,
            'max_bin': 150,
            'bagging_fraction': 0.38914578342083045,
            'bagging_freq': 3,
            'feature_fraction': 0.7984285598881817,
            'min_data_in_leaf': 9,
            'min_sum_hessian_in_leaf': 9
            
        }

        #  ========== データの箱用意 ==========
        self.models = []# 学習済みのモデルを保存
        self.train_test_scores = []# tarin_test_scoreを保存
        self.use_cols = []

    def train(self, X_train, y_train, X_valid, y_valid, X_train_test, y_train_test, categories=None, show_graph=False, print_importance=False, show_importance=False):

        #  train > pred 一周の間のみmodelを保持しておくためmodelsを初期化
        self.models =[]

        #  後のpath名のためにtrainが呼び出された回数をインクリメント
        self.train_counter += 1

        # lightgbm用にデータを変形
        lgb_train = lgb.Dataset(X_train[self.use_cols], y_train)
        lgb_eval = lgb.Dataset(X_valid[self.use_cols], y_valid, reference=lgb_train)

        for s in range(self.random_seed_num):
            self.params["random_seed"]=s
            # モデルの学習
            lgb_results={}
            model_lgb = lgb.train(self.params,
                                    lgb_train,
                                    valid_sets = [lgb_train, lgb_eval],
                                    categorical_feature = categories,
                                    valid_names = ["Train","Test"],
                                    num_boost_round = self.num_boost_round,
                                    early_stopping_rounds = 50,
                                    verbose_eval = 10,
                                    evals_result = lgb_results)
            self.models.append(model_lgb)

            #  valid dataに対する予測
            y_pred = self.predict(X_valid)
            score = np.sqrt(mean_squared_error(y_valid, y_pred))
            print("Lgbm valid rmse: ", score)

            X_valid["pred"] = y_pred
            X_valid["actual"] = y_valid
            X_valid = X_valid.sort_values(["station_id","date","hour"],ascending=True).reset_index(drop=True)

            if show_graph:
                actual_pred_df = pd.DataFrame({
                        "actual" : np.array(X_valid["actual"][0:230]),
                        "pred": np.array(X_valid["pred"][0:230])
                    })
                actual_pred_df = actual_pred_df.reset_index(drop=True)
                actual_pred_df.plot(title="[Valid] score: "+str(score) ,figsize=(18,10), grid=True)
                path = "../figures/model_test2/valid"+str(self.train_counter)+"_seed"+str(s)+".png"
                plt.savefig(path)

            #  train_test dataに対する予測
            y_train_test_pred = self.predict(X_train_test)
            train_test_score = np.sqrt(mean_squared_error(y_train_test, y_train_test_pred))
            self.train_test_scores.append(train_test_score)
            print("Lgbm test rmse: ", train_test_score)

            X_train_test["pred"] = y_train_test_pred
            X_train_test["actual"] = y_train_test
            X_train_test = X_train_test.sort_values(["station_id","date","hour"],ascending=True).reset_index(drop=True)

            if show_graph:
                actual_pred_df = pd.DataFrame({
                        "actual" : np.array(X_train_test["actual"][0:230]),
                        "pred": np.array(X_train_test["pred"][0:230])
                    })
                actual_pred_df = actual_pred_df.reset_index(drop=True)
                actual_pred_df.plot(title="[Train Test] score: "+str(train_test_score) ,figsize=(18,10), grid=True)
                path = "../figures/model_test2/train_test"+str(self.train_counter)+"_seed"+str(s)+".png"
                plt.savefig(path)

            if print_importance:
                print(" Feature Importance ")
                importance = pd.DataFrame(model_lgb.feature_importance(importance_type="gain"), index=X_train[self.use_cols].columns, columns=['importance'])
                print(importance.sort_values('importance', ascending=False))

            if show_importance:
                lgb.plot_importance(model_lgb, importance_type="gain", max_num_features=100)
                plt.show()

    def predict(self, X_test):

        #  dateをhour単位に変更
        X_test['ym_hour'] = X_test['year'].astype(str) + '/' + X_test['month'].astype(str).str.zfill(2).astype(str) + '/' + X_test['day'].astype(str).str.zfill(2).astype(str) + " " + X_test['hour'].astype(str).str.zfill(2).astype(str)
        X_test['ym_hour'] = pd.to_datetime(X_test['ym_hour'])

        preds_array = np.zeros((len(self.models), X_test.shape[0]), dtype=float)

        #  1hourずつ予測 → 次のlag変数を登録
        for i,target_date in enumerate(sorted(list(set(X_test["ym_hour"].to_list())))):

            #  1hour * 70station = 70行
            X_by_date = X_test[(X_test["ym_hour"]==target_date)]

            if(X_by_date.shape[0]!=70):
                print("X_test length not equal 70.")
                exit()

            current_index = 70*i

            for num_model,model in enumerate(self.models):
                #  ランダムシード分回ることになる
                pred = model.predict(X_by_date[self.use_cols], num_iteration=model.best_iteration)
                preds_array[num_model,current_index:current_index+70] = pred

        #  今の所 date, hour, station_id の順番
        pred_array=np.mean(preds_array, axis=0)

        #  dateを日付単位に戻す
        X_test = X_test.drop("ym_hour", axis=1)

        return pred_array



class Rdf_model:
    def __init__(self):

        # ========== パラメータ ==========
        self.train_counter = 0# 後のpath名用
        self.random_seed_num = 1
        self.num_boost_round = 10000# 10000
        self.learning_rate = 0.01
        self.params = {
            "criterion":"mse",
            "max_depth":15,
            "random_state":0,
            "verbose":10
        }

        #  ========== データの箱用意 ==========
        self.models = []# 学習済みのモデルを保存
        self.valid_scores = []
        self.train_test_scores = []# tarin_test_scoreを保存
        self.use_cols = []

    def train(self, X_train, y_train, X_valid, y_valid, X_train_test, y_train_test, categories=None, show_graph=False, print_importance=False, show_importance=False):

        #  train > pred 一周の間のみmodelを保持しておくためmodelsを初期化
        self.models =[]

        #  後のpath名のためにtrainが呼び出された回数をインクリメント
        self.train_counter += 1

        X_train = X_train.fillna(-1)
        X_valid = X_valid.fillna(-1)
        X_train_test = X_train_test.fillna(-1)

        for s in range(self.random_seed_num):
            self.params["random_state"]=s
            # モデルの学習
            model_regr = RandomForestRegressor(**self.params)
            model_regr.fit(X_train[self.use_cols], y_train)
            self.models.append(model_regr)

            #  valid dataに対する予測
            y_pred = self.predict(X_valid)
            score = np.sqrt(mean_squared_error(y_valid, y_pred))
            self.valid_scores.append(score)
            print("Rdf valid rmse: ", score)

            X_valid["pred"] = y_pred
            X_valid["actual"] = y_valid
            X_valid = X_valid.sort_values(["station_id","date","hour"],ascending=True).reset_index(drop=True)

            if show_graph:
                actual_pred_df = pd.DataFrame({
                        "actual" : np.array(X_valid["actual"][0:230]),
                        "pred": np.array(X_valid["pred"][0:230])
                    })
                actual_pred_df = actual_pred_df.reset_index(drop=True)
                actual_pred_df.plot(title="[Valid] score: "+str(score) ,figsize=(18,10), grid=True)
                path = "../figures/model_test2/valid"+str(self.train_counter)+"_seed"+str(s)+".png"
                plt.savefig(path)

            #  train_test dataに対する予測
            y_train_test_pred = self.predict(X_train_test)
            train_test_score = np.sqrt(mean_squared_error(y_train_test, y_train_test_pred))
            self.train_test_scores.append(train_test_score)
            print("Rdf test rmse: ", train_test_score)

            X_train_test["pred"] = y_train_test_pred
            X_train_test["actual"] = y_train_test
            X_train_test = X_train_test.sort_values(["station_id","date","hour"],ascending=True).reset_index(drop=True)

            if show_graph:
                actual_pred_df = pd.DataFrame({
                        "actual" : np.array(X_train_test["actual"][0:230]),
                        "pred": np.array(X_train_test["pred"][0:230])
                    })
                actual_pred_df = actual_pred_df.reset_index(drop=True)
                actual_pred_df.plot(title="[Train Test] score: "+str(train_test_score) ,figsize=(18,10), grid=True)
                path = "../figures/model_test2/train_test"+str(self.train_counter)+"_seed"+str(s)+".png"
                plt.savefig(path)

    def predict(self, X_test):

        X_test = X_test.fillna(-1)

        #  dateをhour単位に変更
        X_test['ym_hour'] = X_test['year'].astype(str) + '/' + X_test['month'].astype(str).str.zfill(2).astype(str) + '/' + X_test['day'].astype(str).str.zfill(2).astype(str) + " " + X_test['hour'].astype(str).str.zfill(2).astype(str)
        X_test['ym_hour'] = pd.to_datetime(X_test['ym_hour'])

        preds_array = np.zeros((len(self.models), X_test.shape[0]), dtype=float)

        #  1hourずつ予測 → 次のlag変数を登録
        for i,target_date in enumerate(sorted(list(set(X_test["ym_hour"].to_list())))):

            #  1hour * 70station = 70行
            X_by_date = X_test[(X_test["ym_hour"]==target_date)]

            if(X_by_date.shape[0]!=70):
                print("X_test length not equal 70.")
                exit()

            current_index = 70*i

            for num_model,model in enumerate(self.models):
                #  ランダムシード分回ることになる
                pred = model.predict(X_by_date[self.use_cols])
                preds_array[num_model,current_index:current_index+70] = pred

                #  lag変数の更新
                #  1h~3hlag変数
                for n_shift in range(3):

                    col_name = "bikes_available_"+str(n_shift+1)+"hour_lag"
                    shifted_index = current_index + 70*(n_shift+1)

                    if(((i%23)+1)==23):
                        continue

                    elif(((i%23)+1)==22):
                        #  +1 無し +2
                        if n_shift<1:
                            if n_shift==2:
                                shifted_index -= 70
                        else:
                            continue

                    elif(((i%23)+1)==21):
                        if n_shift<2:
                            #  +1 +2 無し
                            pass
                        else:
                            continue

                    if(X_test.shape[0]>shifted_index):
                        X_test[shifted_index:shifted_index+70][col_name] = pred

        #  今の所 date, hour, station_id の順番
        pred_array=np.mean(preds_array, axis=0)

        #  dateを日付単位に戻す
        X_test = X_test.drop("ym_hour", axis=1)

        return pred_array



if __name__ == '__main__':

    #  カテゴリカルデータ指定
    categories = ["week_num","station_id","CE_city","CE_events"]

    #  全データ共通の前処理後のデータ 取得・表示
    all_df = pd.read_csv("../input/all_df.csv")
    print("====== columns =====")
    print(all_df.columns)
    print("====================")

    #  学習
    model = Lgb_model()
    model.train(all_df, categories)

