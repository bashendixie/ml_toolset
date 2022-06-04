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

week_day_dict = {
    'Streptococcus_pyogenes' : 0,
    'Salmonella_enterica' : 1,
    'Enterococcus_hirae' : 2,
    'Escherichia_coli' : 3,
    'Campylobacter_jejuni' : 4,
    'Streptococcus_pneumoniae' : 5,
    'Staphylococcus_aureus' : 6,
    'Escherichia_fergusonii' : 7,
    'Bacteroides_fragilis' : 8,
    'Klebsiella_pneumoniae' : 9
}


data1 = pd.read_csv('withtarget/train.csv')
data2 = pd.read_csv('withtarget/test.csv')
# 合并
data3 = pd.concat([data1, data2])
# 删除id列
del data3['row_id']

data3.drop_duplicates(keep='first', inplace=True)
print(data3.shape)

# 获取标签
labels = data3['target']
# 删除标签列
del data3['target']
print(data3.shape)
print(labels.shape)

labels_arr = []
for lab in labels:
    labels_arr.append(week_day_dict[lab])

# 初始化模型
estimators = [
    ('AdaBoostClassifier', RandomForestClassifier (n_estimators=200)),
    ('Random Forest', ExtraTreesClassifier(random_state=42, n_estimators=200)),
    ('Extra Trees', AdaBoostClassifier(n_estimators=500))
]

random = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=4)
random.fit(data3, labels_arr)

# 保存模型
s=pickle.dumps(random)
f=open('StackingClassifier_v3.model', "wb+")
f.write(s)
f.close()