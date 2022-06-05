from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, mean_squared_error,mean_absolute_error, f1_score
import numpy as np
import pandas as pd
from statistics import mean
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import ExtraTreesRegressor
import pickle
from sklearn.ensemble import BaggingRegressor


train = pd.read_csv('group/merge.csv', index_col='id')
test = pd.read_csv('origin/test.csv', index_col='id')
ss = pd.read_csv('origin/sample_submission.csv')

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

y = train['target']
del train['target']
# del test['cont9']

train['max'] = train.max(axis=1)
train['min'] = train.min(axis=1)
train['mean'] = train.mean(axis=1)
train['sum'] = train.sum(axis=1)
train['cha'] = train.max(axis=1) - train.min(axis=1)
train['zhong'] = (train.max(axis=1) + train.min(axis=1))/2

test['max'] = test.max(axis=1)
test['min'] = test.min(axis=1)
test['mean'] = test.mean(axis=1)
test['sum'] = test.sum(axis=1)
test['cha'] = test.max(axis=1) - test.min(axis=1)
test['zhong'] = (test.max(axis=1) + test.min(axis=1))/2

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


def train1():
    regr = BaggingRegressor(n_estimators=100, oob_score=True, random_state=1010)
    regr.fit(train, y)

    # 保存模型
    s=pickle.dumps(regr)
    f=open('BaggingRegressor_v8.model', "wb+")
    f.write(s)
    f.close()

def test1():
    print("这是BaggingRegressor")
    f2 = open('BaggingRegressor_v8.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)

    predictions = model1.predict(test)
    preds = []
    for pred in predictions:
        preds.append(pred)

    res = pd.DataFrame()
    res['target'] = preds
    res.to_csv("predict_BaggingRegressor_v8.csv")

test1()
