from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import StackingClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier

# X = pd.read_csv('data/train_data.csv')
# X.drop_duplicates(keep='first', inplace=True)
# duplicates_train = X.duplicated().sum()
# print('Duplicates in train data: {0}'.format(duplicates_train))
#
# label = pd.read_csv('data/train_label.csv')
# y = label.target

data = pd.read_csv('data/train_for_duplicated_data.csv')

# 去除重复
duplicates_train = data.duplicated().sum()
print('Duplicates in train data: {0}'.format(duplicates_train))

data.drop_duplicates(keep='first', inplace=True)
duplicates_train = data.duplicated().sum()

print('Train data shape:', data.shape)
print('Duplicates in train data: {0}'.format(duplicates_train))

# 获取标签列
labels = data['target']

# 删除标签列
del data['target']

# 初始化模型
bag_clf=BaggingClassifier(ExtraTreesClassifier(),
		n_estimators=400,
		max_samples=10000,
		bootstrap=True,
		n_jobs=4,
		oob_score=True)

bag_clf.fit(data, labels)
print(bag_clf.oob_score)

# 保存模型
s=pickle.dumps(bag_clf)
f=open('BaggingClassifier_v5.model', "wb+")
f.write(s)
f.close()