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


def conv():
    print(22222222222)
    model = load_model("tabular_conv_1.h5")
    model.summary()

    tests = pd.read_csv('data/test_k.csv')
    tests = np.expand_dims(tests.values.astype(float), axis=2)

    predictions = model.predict(tests, batch_size=6570)
    print(predictions)

    preds = []
    for pred in predictions:
        preds.append(pred[0])

    res = pd.DataFrame()
    res['num'] = preds
    res.to_csv("tabular_conv_v1.csv")

def test0():
    print(1111111111111111)

#
# def test1():
#     model = load_model("tabular_fnn_2.h5")
#     model.summary()
#
#     test = pd.read_csv('data/test_fnn.csv')
#     #cols = ['date', 'Weekday', 'Festival', 'country_Finland', 'country_Norway', 'country_Sweden', 'store_KaggleMart', 'store_KaggleRama', 'product_Kaggle Hat', 'product_Kaggle Mug', 'product_Kaggle Sticker']
#     cols = ['date', 'week', 'festival', 'country', 'store', 'product']
#     tests = test[cols]#.head()
#     #tests = test.to_numpy()
#     #tests = tests.astype('float64')
#
#     predictions = model.predict(tests, batch_size=6570)
#     print(predictions)
#
#     test['num'] = predictions
#     test.to_csv("predict_v15.csv")
#
#
# def test2():
#     model = load_model("tabular_13_3.h5")
#     model.summary()
#
#     test = pd.read_csv('data/test_k.csv')
#     # cols = ['date', 'Weekday', 'Festival', 'country_Finland', 'country_Norway', 'country_Sweden', 'store_KaggleMart', 'store_KaggleRama', 'product_Kaggle Hat', 'product_Kaggle Mug', 'product_Kaggle Sticker']
#     #cols = ['date', 'week', 'festival', 'country', 'store', 'product']
#     #tests = test[cols]  # .head()
#
#     tests = test.to_numpy()
#     tests = tests.astype('float64')
#
#     predictions = model.predict(tests, batch_size=6570)
#     print(predictions)
#
#     test['num'] = predictions
#     test.to_csv("predict_v16.csv")

def test_for_extratrees():
    print("这是ExtraTreesRegressor")
    f2 = open('ExtraTreesRegressor.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)
    test_X = pd.read_csv('data/test_k.csv')

    predictions = model1.predict(test_X)
    preds = []
    for pred in predictions:
        preds.append(pred)

    res = pd.DataFrame()
    res['num_sold'] = preds
    res.to_csv("predict_ExtraTreesRegressor_v1.csv")



def test_for_StackingRegressor():
    print("这是StackingRegressor")
    f2 = open('StackingRegressor.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)
    test_X = pd.read_csv('data/test_k.csv')

    predictions = model1.predict(test_X)
    preds = []
    for pred in predictions:
        preds.append(pred)

    res = pd.DataFrame()
    res['num_sold'] = preds
    res.to_csv("predict_StackingRegressor_v1.csv")

test_for_StackingRegressor()










