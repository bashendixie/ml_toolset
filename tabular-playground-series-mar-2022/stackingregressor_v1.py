from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV


train = pd.read_csv('child/train_action.csv', index_col='row_id')
test = pd.read_csv('child/test_action.csv', index_col='row_id')
ss = pd.read_csv('origin/sample_submission.csv')

def feature_engineering(data):
    data['time'] = pd.to_datetime(data['time'])
    data['month'] = data['time'].dt.month
    data['weekday'] = data['time'].dt.weekday
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    data['is_month_start'] = data['time'].dt.is_month_start.astype('int')
    data['is_month_end'] = data['time'].dt.is_month_end.astype('int')
    data['hour+minute'] = data['time'].dt.hour * 60 + data['time'].dt.minute
    data['is_weekend'] = (data['time'].dt.dayofweek > 4).astype('int')
    data['is_afternoon'] = (data['time'].dt.hour > 12).astype('int')
    data['x+y'] = data['x'].astype('str') + data['y'].astype('str')
    data['x+y+direction'] = data['x'].astype('str') + data['y'].astype('str') + data['direction'].astype('str')
    data['hour+direction'] = data['hour'].astype('str') + data['direction'].astype('str')
    data['hour+x+y'] = data['hour'].astype('str') + data['x'].astype('str') + data['y'].astype('str')
    data['hour+direction+x'] = data['hour'].astype('str') + data['direction'].astype('str') + data['x'].astype('str')
    data['hour+direction+y'] = data['hour'].astype('str') + data['direction'].astype('str') + data['y'].astype('str')
    data['hour+direction+x+y'] = data['hour'].astype('str') + data['direction'].astype('str') + data['x'].astype('str') + data['y'].astype('str')
    data['hour+x'] = data['hour'].astype('str') + data['x'].astype('str')
    data['hour+y'] = data['hour'].astype('str') + data['y'].astype('str')
    data = data.drop(['time'], axis=1)
    return data

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

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

for data in [train, test]:
    data = feature_engineering(data)

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

for data in [train, test]:
    data = feature_engineering(data)

y = train['congestion']
del train['time']
#del train['time1']
del train['congestion']

del test['time']
#del test['time1']
del test['pre']

def train1():
    # 初始化模型
    estimators = [
        ('Random Forest', RandomForestRegressor(n_estimators=200, random_state=42)),
        ('Lasso', LassoCV(n_alphas=200)),
        ('Gradient Boosting', HistGradientBoostingRegressor(max_iter=200, random_state=0)),
        ('ExtraTrees', ExtraTreesRegressor(n_estimators=200))
    ]

    random = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
    random.fit(train, y)

    # 保存模型
    s=pickle.dumps(random)
    f=open('StackingRegressor_v1.model', "wb+")
    f.write(s)
    f.close()

def test1():
    print("这是StackingRegressor")
    f2 = open('StackingRegressor_v1.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)

    predictions = model1.predict(test)
    preds = []
    for pred in predictions:
        preds.append(pred)

    res = pd.DataFrame()
    res['congestion'] = preds
    res.to_csv("predict_StackingRegressor_v1.csv")

test1()