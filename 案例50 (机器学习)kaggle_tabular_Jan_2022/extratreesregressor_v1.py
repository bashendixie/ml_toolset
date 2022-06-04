from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import ExtraTreesRegressor


data = pd.read_csv('data/train_k_data.csv')
label = pd.read_csv('data/train_k_label.csv')
y = label.num_sold

random = ExtraTreesRegressor(n_estimators=100)
random.fit(data,y)

# 保存模型
s=pickle.dumps(random)
f=open('ExtraTreesRegressor.model', "wb+")
f.write(s)
f.close()