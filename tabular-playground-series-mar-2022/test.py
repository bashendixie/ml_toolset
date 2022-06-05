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

def feature_engineering(data):
    data['time'] = pd.to_datetime(data['time'])
    data['month'] = data['time'].dt.month
    data['weekday'] = data['time'].dt.weekday
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    data['is_month_start'] = data['time'].dt.is_month_start.astype('int')
    data['is_month_end'] = data['time'].dt.is_month_end.astype('int')
    data['hour+minute'] = data['time'].dt.hour * 60 + data['time'].dt.minute
    data['is_weekend'] = (data['time'].dt.dayofweek > 4).astype('int')
    data['is_afternoon'] = (data['time'].dt.hour > 12).astype('int')
    data['x+y'] = data['x'].astype('str') + data['y'].astype('str')
    data['x+y+direction'] = data['x'].astype('str') + data['y'].astype('str') + data['direction'].astype('str')
    data['hour+direction'] = data['hour'].astype('str') + data['direction'].astype('str')
    data['hour+x+y'] = data['hour'].astype('str') + data['x'].astype('str') + data['y'].astype('str')
    data['hour+direction+x'] = data['hour'].astype('str') + data['direction'].astype('str') + data['x'].astype('str')
    data['hour+direction+y'] = data['hour'].astype('str') + data['direction'].astype('str') + data['y'].astype('str')
    data['hour+direction+x+y'] = data['hour'].astype('str') + data['direction'].astype('str') + data['x'].astype('str') + data['y'].astype('str')
    data['hour+x'] = data['hour'].astype('str') + data['x'].astype('str')
    data['hour+y'] = data['hour'].astype('str') + data['y'].astype('str')
    data = data.drop(['time'], axis=1)
    return data

def test_for_extratrees():
    print("这是ExtraTreesRegressor1")
    f2 = open('ExtraTreesRegressor_v3.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)
    test_X = pd.read_csv('test_action.csv')
    del test_X['row_id']
    del test_X['time']
    test_X['time1'] = test_X['time1'] - 670435200

    predictions = model1.predict(test_X)
    preds = []
    for pred in predictions:
        preds.append(pred)

    res = pd.DataFrame()
    res['num_sold'] = preds
    res.to_csv("predict_ExtraTreesRegressor_v3.csv")

def test_for_extratrees_v1():
    print("这是ExtraTreesRegressor2")
    f2 = open('ExtraTreesRegressor_v1.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)
    test_X = pd.read_csv('test_action.csv', index_col='row_id', parse_dates=['time'])

    for data in [test_X]:
        data = feature_engineering(data)

    del test_X['time']

    predictions = model1.predict(test_X)
    preds = []
    for pred in predictions:
        preds.append(pred)

    res = pd.DataFrame()
    res['num_sold'] = preds
    res.to_csv("predict_ExtraTreesRegressor_v1.csv")

test_for_extratrees_v1()










