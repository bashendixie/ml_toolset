import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D,Conv1D,Embedding, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

data = pd.read_csv('data/train_for_duplicated_data.csv')

# 去除重复
duplicates_train = data.duplicated().sum()
print('Duplicates in train data: {0}'.format(duplicates_train))

data.drop_duplicates(keep='first', inplace=True)
duplicates_train = data.duplicated().sum()

print('Train data shape:', data.shape)
print('Duplicates in train data: {0}'.format(duplicates_train))

# 获取标签列
labels = data['target']

# 删除标签列
del data['target']

# 归一化
data = (data - data.min()) / (data.max() - data.min())
# 增加和、平均、最大、最小的列
res = pd.DataFrame()
res['sum'] = data.sum(axis=1)
res['max'] = data.max(axis=1)
res['min'] = data.min(axis=1)
res['mean'] = data.mean(axis=1)
res['var'] = data.var(axis=1)
res['std'] = data.std(axis=1)

data['sum'] = res['sum']
data['max'] = res['max']
data['min'] = res['min']
data['mean'] = res['mean']
data['var'] = res['var']
data['std'] = res['std']

# 打乱数据，先注释掉，之前是打乱了训练数据，但是标签错误，真是蠢
# data.sample(frac=1)

print(data.head())
trains = data.to_numpy()#.head()

one_hot_train_labels = to_categorical(labels)

#trains = trains.astype('float64')
#labels = labels.astype('float64')


# 构建模型
def build_model():
    model = Sequential([
        Dense(1024, activation=tf.nn.relu, input_shape=[292,]),
        Dense(512, activation=tf.nn.relu),
        Dense(256, activation=tf.nn.relu),
        Dense(128, activation=tf.nn.relu),
        Dense(64, activation=tf.nn.relu),
        Dense(32, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=optimizers.RMSprop(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class PrintDot(callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print(epoch)
        #print(logs)
        print('.', end='')

# 构建模型
model = build_model()
#model = load_model("tabular_fnn_dup_1.h5")
#model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])#learning_rate=0.0005

# 当监测数量停止改善时停止训练
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=50)
save_func = callbacks.ModelCheckpoint(filepath='tabular_fnn_dup_addlie_1.h5', save_best_only=True)

# 训练模型
H = model.fit(trains, one_hot_train_labels, epochs=500, verbose=1, validation_split=0.5, callbacks=[early_stop, PrintDot(), save_func])

