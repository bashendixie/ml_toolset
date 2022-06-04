from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV


data = pd.read_csv('data/train_k_data.csv')
label = pd.read_csv('data/train_k_label.csv')
y = label.num_sold

# 初始化模型
estimators = [
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('Lasso', LassoCV()),
    ('Gradient Boosting', HistGradientBoostingRegressor(random_state=0))
]

random = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
random.fit(data,y)

# 保存模型
s=pickle.dumps(random)
f=open('StackingRegressor.model', "wb+")
f.write(s)
f.close()