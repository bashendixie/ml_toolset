from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import ExtraTreesRegressor, BaggingClassifier
import time, datetime

def str2date(str, date_format="%Y/%m/%d %H:%M"):
  date = datetime.datetime.strptime(str, date_format)
  return date

# Read the data
# train = pd.read_csv('origin/train.csv', index_col='row_id', parse_dates=['time'])
test = pd.read_csv('origin/test.csv', index_col='row_id', parse_dates=['time'])
train = pd.read_csv('train_action_v1.csv', index_col='row_id', parse_dates=['time'])

# Feature Engineering
for df in [train, test]:
    df['weekday'] = df.time.dt.weekday
    df['hour'] = df.time.dt.hour
    df['minute'] = df.time.dt.minute

medians = train.groupby(['x', 'y', 'direction', 'weekday', 'hour', 'minute']).congestion.median().astype(int)
print(medians)

means = train.groupby(['x', 'y', 'direction', 'weekday', 'hour', 'minute']).congestion.mean().astype(int)
print(means)

maxs = train.groupby(['x', 'y', 'direction', 'weekday', 'hour', 'minute']).congestion.max().astype(int)
print(maxs)

mins = train.groupby(['x', 'y', 'direction', 'weekday', 'hour', 'minute']).congestion.min().astype(int)
print(mins)

halfs = (medians + means) / 2
print(halfs)

temp = train
del temp['congestion']
sub = temp.merge(halfs, left_on=['x', 'y', 'direction', 'weekday', 'hour', 'minute'], right_index=True)[['congestion']]
sub.reset_index(inplace=True)
y = sub['congestion']
del train['time']

print(train.head())
print(y.head())

random = ExtraTreesRegressor(n_estimators=100)
random.fit(train, y)

# 保存模型
s=pickle.dumps(random)
f=open('ExtraTreesRegressor_v3.model', "wb+")
f.write(s)
f.close()