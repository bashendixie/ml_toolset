from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, mean_squared_error,mean_absolute_error, f1_score
import numpy as np
import pandas as pd
from statistics import mean
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import ExtraTreesRegressor


train = pd.read_csv('del.csv', index_col='row_id')
test = pd.read_csv('test_action.csv', index_col='row_id')
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

y = train['congestion']
del train['time']
#del train['time1']
del train['congestion']

del test['time']
del test['time1']
#del test['pre']

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
    predictions_etr_263 += model.predict(test[test.columns.tolist()[0:-1]]) / 10

print("CV score: {:<8.8f}".format(mean_absolute_error(oof_etr_263, y)))

res = pd.DataFrame()
res['congestion'] = predictions_etr_263
res.to_csv('submission_v103.csv', index=False)