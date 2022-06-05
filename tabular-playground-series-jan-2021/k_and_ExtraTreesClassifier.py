from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, mean_squared_error,mean_absolute_error, f1_score
import numpy as np
import pandas as pd
from statistics import mean
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import ExtraTreesRegressor




def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col!= 'time':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def train1():
    train = pd.read_csv('origin/train.csv', index_col='id')
    test = pd.read_csv('origin/test.csv', index_col='id')
    ss = pd.read_csv('origin/sample_submission.csv')

    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    y = train['target']
    del train['target']

    features = train.columns.tolist()[0:-1]
    X = train[features]

    #分类问题用 StrtifiedKFold
    #回归问题用 KFold
    #集成学习（下）实战案例一——幸福感预测
    #https://www.icode9.com/content-4-978427.html

    predictions, scores = [], []
    oof_etr_263 = np.zeros(train.shape[0])
    predictions_etr_263 = np.zeros(len(test))
    k = KFold(n_splits=20, random_state=228, shuffle=True)
    for i, (trn_idx, val_idx) in enumerate(k.split(X, y)):
        print("第一次：" + str(i))
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        model = ExtraTreesRegressor(n_estimators=200, n_jobs=-1)
        model.fit(X_train, y_train)
        oof_etr_263[val_idx] = model.predict(X_val)
        predictions_etr_263 += model.predict(test[test.columns.tolist()[0:-1]]) / 20

    print("CV score: {:<8.8f}".format(mean_absolute_error(oof_etr_263, y)))

    res = pd.DataFrame()
    res['target'] = predictions_etr_263
    res.to_csv('submission_k_and_extra_v1.csv', index=False)


def train2():
    train = pd.read_csv('group/merge.csv', index_col='id')
    test = pd.read_csv('origin/test.csv', index_col='id')

    y = train['target']
    del train['target']

    train['max'] = train.max(axis=1)
    train['min'] = train.min(axis=1)
    train['mean'] = train.mean(axis=1)
    train['sum'] = train.sum(axis=1)
    train['cha'] = train.max(axis=1) - train.min(axis=1)
    train['zhong'] = (train.max(axis=1) + train.min(axis=1)) / 2

    test['max'] = test.max(axis=1)
    test['min'] = test.min(axis=1)
    test['mean'] = test.mean(axis=1)
    test['sum'] = test.sum(axis=1)
    test['cha'] = test.max(axis=1) - test.min(axis=1)
    test['zhong'] = (test.max(axis=1) + test.min(axis=1)) / 2

    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    features = train.columns.tolist()[0:-1]
    X = train[features]

    # 分类问题用 StrtifiedKFold
    # 回归问题用 KFold
    # 集成学习（下）实战案例一——幸福感预测
    # https://www.icode9.com/content-4-978427.html

    predictions, scores = [], []
    oof_etr_263 = np.zeros(train.shape[0])
    predictions_etr_263 = np.zeros(len(test))
    k = KFold(n_splits=20, random_state=228, shuffle=True)
    for i, (trn_idx, val_idx) in enumerate(k.split(X, y)):
        print("第一次：" + str(i))
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        model = ExtraTreesRegressor(n_estimators=200, n_jobs=-1)
        model.fit(X_train, y_train)
        oof_etr_263[val_idx] = model.predict(X_val)
        predictions_etr_263 += model.predict(test[test.columns.tolist()[0:-1]]) / 20

    print("CV score: {:<8.8f}".format(mean_absolute_error(oof_etr_263, y)))

    res = pd.DataFrame()
    res['target'] = predictions_etr_263
    res.to_csv('submission_k_and_extra_v2.csv', index=False)

train2()