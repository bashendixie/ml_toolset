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
from tensorflow.keras.layers import Dense
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import backend as K

data = pd.read_csv('data/train_k_data.csv')
label = pd.read_csv('data/train_k_label.csv')
#cols = ['date', 'Weekday', 'Festival', 'country_Finland', 'country_Norway', 'country_Sweden', 'store_KaggleMart', 'store_KaggleRama', 'product_Kaggle Hat', 'product_Kaggle Mug', 'product_Kaggle Sticker']

trains = data.to_numpy()#.head()
labels = label.num_sold

trains = trains.astype('float64')
labels = labels.astype('float64')

print(trains.shape)

# 构建模型
def build_model():
    model = Sequential([
        Dense(170, activation=tf.nn.relu, input_shape=[170]),
        Dense(1024, activation=tf.nn.relu),
        Dense(512, activation=tf.nn.relu),
        Dense(256, activation=tf.nn.relu),
        Dense(128, activation=tf.nn.relu),
        Dense(64, activation=tf.nn.relu),
        Dense(32, activation=tf.nn.relu),
        Dense(16, activation=tf.nn.relu),
        Dense(8, activation=tf.nn.relu),
        Dense(1)
    ])

    #model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
    model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
    return model


def baseline_model():
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=1, strides=1, input_shape=(170, 1)))
    model.add(Flatten())
    #model.add(Dense(units=25, input_dim=(50,), activation='relu'))
    model.add(Dense(256, activation=tf.nn.tanh))#tf.nn.relu
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(1))
    model.compile(optimizer=optimizers.SGD(), loss='mse', metrics=['mae', 'mse'])
    return model

class PrintDot(callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print(epoch)
        #print(logs)
        print('.', end='')

# 构建模型
#model = build_model()
#model = baseline_model()
model = load_model("tabular_13_2.h5")
# 当监测数量停止改善时停止训练
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=50)
save_func = callbacks.ModelCheckpoint(filepath='tabular_13_3.h5', save_best_only=True)

# 训练模型
H = model.fit(trains, labels, epochs=500, verbose=1, validation_split = 0.1,
                    callbacks=[PrintDot(), save_func])#early_stop,

#model.save("tabular_13_1.h5")

# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, 20), H.history["loss"], masks="train_loss")
# plt.plot(np.arange(0, 20), H.history["val_loss"], masks="val_loss")
# plt.plot(np.arange(0, 20), H.history["accuracy"], masks="train_acc")
# plt.plot(np.arange(0, 20), H.history["val_accuracy"], masks="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.show()

