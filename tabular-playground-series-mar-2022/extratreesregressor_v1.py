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



data = pd.read_csv('train_action.csv')
del data['row_id']
del data['time']
data['time1'] = data['time1'] - 670435200

y = data['congestion']
del data['congestion']

print(data.head())
print(y.head())

random = ExtraTreesRegressor(n_estimators=150)
random.fit(data, y)

# 保存模型
s=pickle.dumps(random)
f=open('ExtraTreesRegressor_v3.model', "wb+")
f.write(s)
f.close()