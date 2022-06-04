from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import ExtraTreesClassifier

# data = pd.read_csv('data/train_for_duplicated_data.csv')
# data.drop_duplicates(keep='first', inplace=True)
# # 获取标签列
# labels = data['target']
# # 删除标签列
# del data['target']


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

#打乱数据
data3.sample(frac=1)
#打乱数据
data3.sample(frac=1)
#打乱数据
data3.sample(frac=1)

# 获取标签
labels = data3['target']
# 删除标签列
del data3['target']
print(data3.shape)
print(labels.shape)

labels_arr = []
for lab in labels:
    labels_arr.append(week_day_dict[lab])

random = ExtraTreesClassifier(n_estimators=1000, n_jobs=4)
random.fit(data3, labels_arr)

# 保存模型
s=pickle.dumps(random)
f=open('extra_trees_train_and_test_dup_v1.model', "wb+")
f.write(s)
f.close()