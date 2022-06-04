from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

X = pd.read_csv('data/train_data.csv')
label = pd.read_csv('data/train_label.csv')
y = label.target

random = RandomForestClassifier(n_estimators=100)
random.fit(X,y)

# 保存模型
s=pickle.dumps(random)
f=open('random_forest.model', "wb+")
f.write(s)
f.close()