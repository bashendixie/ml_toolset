from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
# 1. Using XGBoost Classifier
import xgboost as xgb
from xgboost import XGBClassifier


X = pd.read_csv('data/train_data.csv')
label = pd.read_csv('data/train_label.csv')
y = label.target

# creating dataset split for prediction
from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(X, y,test_size = 0.2,random_state = 42) # 80-20 split

# Checking split
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)


# from sklearn.preprocessing import LabelEncoder
# enc = LabelEncoder()
# y = enc.fit_transform(y)

model1 = XGBClassifier(random_state = 42,n_estimators = 1000,learning_rate = 0.05,eval_metric ="auc",
                      max_depth = 8,booster ='gbtree',verbosity=0,tree_method = 'gpu_hist',task_type ="GPU")
model1.fit(X,y)


# 保存模型
s=pickle.dumps(model1)
f=open('xgboost.model', "wb+")
f.write(s)
f.close()