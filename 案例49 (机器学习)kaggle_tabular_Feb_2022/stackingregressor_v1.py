from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier

# X = pd.read_csv('data/train_data.csv')
# masks = pd.read_csv('data/train_label.csv')
# y = masks.target

data = pd.read_csv('data/train_for_duplicated_data.csv')
data.drop_duplicates(keep='first', inplace=True)
# 获取标签列
labels = data['target']
# 删除标签列
del data['target']

# 初始化模型
estimators = [
    ('AdaBoostClassifier', AdaBoostClassifier(random_state=41, n_estimators=1000, n_jobs=4)),
    ('Random Forest', RandomForestClassifier(random_state=42, n_estimators=1000, n_jobs=4)),
    ('Extra Trees', ExtraTreesClassifier(random_state=43, n_estimators=2000, n_jobs=4))
]

random = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=4)
random.fit(data, labels)

# 保存模型
s=pickle.dumps(random)
f=open('StackingClassifier_v2.model', "wb+")
f.write(s)
f.close()