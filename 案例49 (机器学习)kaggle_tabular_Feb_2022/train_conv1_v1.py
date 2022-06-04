
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
    model.add(Conv1D(16, 3, input_shape=(286, 1), activation='relu'))
    model.add(Conv1D(16, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(10, activation=tf.nn.sigmoid))#, activation='linear'
    #plot_model(model, to_file='./model_linear.png', show_shapes=True)
    print(model.summary())
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


data = pd.read_csv('data/train_data.csv')
label = pd.read_csv('data/train_label.csv')
#cols = ['date', 'Weekday', 'Festival', 'country_Finland', 'country_Norway', 'country_Sweden', 'store_KaggleMart', 'store_KaggleRama', 'product_Kaggle Hat', 'product_Kaggle Mug', 'product_Kaggle Sticker']

#data = data * 1000000
#data = data - (data.max() - 1)
data = (data - data.min()) / (data.max() - data.min())
#打乱数据
data.sample(frac=1)

print(data.head())
trains = np.expand_dims(data.values.astype(float), axis=2)
print(trains.shape)
#trains = data.to_numpy()#.head()

labels = label.target
one_hot_train_labels = to_categorical(labels)

#model = build_model()
model = load_model('tabular_conv_1.h5')
model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# 当监测数量停止改善时停止训练
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=50)
save_func = callbacks.ModelCheckpoint(filepath='tabular_conv_2.h5', save_best_only=True)

# 训练模型
H = model.fit(trains, one_hot_train_labels, epochs=500, verbose=1, validation_split=0.5, callbacks=[early_stop, save_func])#PrintDot(),