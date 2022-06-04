from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import ExtraTreesClassifier

X = pd.read_csv('data/train_data.csv')
label = pd.read_csv('data/train_label.csv')
y = label.target

random = ExtraTreesClassifier(n_estimators=1000)
random.fit(X, y)

# 保存模型
s=pickle.dumps(random)
f=open('extra_trees_v2.model', "wb+")
f.write(s)
f.close()