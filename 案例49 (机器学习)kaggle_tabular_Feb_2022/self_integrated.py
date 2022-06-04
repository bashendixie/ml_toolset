import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


a = pd.read_csv('predict_result/0.93745/predict_xgboost_v1.csv')
b = pd.read_csv('predict_result/0.95341/predict_gcforest_v1.csv')
c = pd.read_csv('predict_result/0.96847/predict_deepforest_v1.csv')

a_1 = a.target
b_1 = b.target
c_1 = c.target

preds = []
for i in range(len(a_1)):
    if a_1[i] == b_1[i]:
        preds.append(a_1[i])
    elif c_1[i] == b_1[i]:
        preds.append(b_1[i])
    elif c_1[i] == a_1[i]:
        preds.append(a_1[i])
    else:
        preds.append(c_1[i])

res = pd.DataFrame()
res['target'] = preds
res.to_csv("predict_integrated_v1.csv")