from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

data = pd.read_csv('data/train_data.csv')
label = pd.read_csv('data/train_label.csv')
data = (data - data.min()) / (data.max() - data.min())
#打乱数据
#data.sample(frac=1)
#print(data.head())
trains = data.to_numpy()
labels = label.target

x_train, x_test, y_train, y_test = train_test_split(trains, labels, random_state=1, train_size=0.9)

clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())
# 保存模型
s=pickle.dumps(clf)
f=open('svm_v3.model', "wb+")
f.write(s)
f.close()

#print(clf.score(x_train, y_train))  # 精度
#y_hat = clf.predict(x_train)
#show_accuracy(y_hat, y_train, '训练集')

#print(clf.score(x_test, y_test))
#y_hat = clf.predict(x_test)
#show_accuracy(y_hat, y_test, '测试集')
