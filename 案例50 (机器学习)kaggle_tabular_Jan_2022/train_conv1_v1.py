import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D,Conv1D,Embedding
from tensorflow.keras.layers import MaxPooling2D, MaxPooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,train_test_split

# 构建模型
def build_model():
    model = Sequential()
    model.add(Conv1D(16, 3, input_shape=(170, 1), activation='relu'))
    model.add(Conv1D(16, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(3, padding='same'))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
    return model

data = pd.read_csv('data/train_k_data.csv')
trains = np.expand_dims(data.values.astype(float), axis=2)

label = pd.read_csv('data/train_k_label.csv')
labels = label.num_sold


model = build_model()

# 当监测数量停止改善时停止训练
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=50)
save_func = callbacks.ModelCheckpoint(filepath='tabular_conv_1.h5', save_best_only=True)

# 训练模型
H = model.fit(trains, labels, epochs=500, verbose=1, validation_split=0.1, callbacks=[early_stop, save_func])