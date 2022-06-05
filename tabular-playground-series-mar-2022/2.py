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


def test_for_extratrees():
    print("这是ExtraTreesRegressor")
    f2 = open('ExtraTreesRegressor_v3.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)
    test = pd.read_csv('test_action_v1.csv', index_col='row_id', parse_dates=['time'])

    for df in [test]:
        df['weekday'] = df.time.dt.weekday
        df['hour'] = df.time.dt.hour
        df['minute'] = df.time.dt.minute

    #del test['row_id']
    del test['time']

    predictions = model1.predict(test)
    preds = []
    for pred in predictions:
        preds.append(pred)

    res = pd.DataFrame()
    res['num_sold'] = preds
    res.to_csv("predict_ExtraTreesRegressor_v3.csv")

test_for_extratrees()










