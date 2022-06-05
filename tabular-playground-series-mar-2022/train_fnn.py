import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D,Conv1D,Embedding
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import backend as K


train = pd.read_csv('child/train_action.csv', index_col='row_id')
test = pd.read_csv('child/test_action.csv', index_col='row_id')

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

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col!= 'time':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

for data in [train, test]:
    data = feature_engineering(data)

y = train['congestion']
del train['time']
del train['congestion']

del test['time']
del test['pre']

features = train.columns.tolist()[0:-1]
X = train[features].astype(np.float32)
y.astype(np.float32)

test = test[features].astype(np.float32)

# 构建模型
def build_model():
    model = Sequential([
        BatchNormalization(input_dim=20),
        Dense(160, activation=tf.nn.relu),
        Dropout(0.1),
        Dense(80, activation=tf.nn.relu),
        Dropout(0.1),
        Dense(40, activation=tf.nn.relu),
        Dropout(0.1),
        Dense(10, activation=tf.nn.relu),
        Dense(1)
    ])
    model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
    return model

class PrintDot(callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print(epoch)
        #print(logs)
        print('.', end='')

def train():
    # 构建模型
    model = build_model()
    #model = load_model("tabular_fnn_1.h5")
    # 当监测数量停止改善时停止训练
    save_func = callbacks.ModelCheckpoint(filepath='tabular_fnn_1.h5', save_best_only=True)

    # 训练模型
    H = model.fit(X, y, epochs=10000, verbose=1, validation_split = 0.1, callbacks=[save_func])

def test1():
    model = load_model("tabular_fnn_1.h5")
    predictions = model.predict(test)
    preds = []
    for pred in predictions:
        preds.append(pred[0])

    res = pd.DataFrame()
    res['num_sold'] = preds
    res.to_csv("predict_fnn_v1.csv")

test1()