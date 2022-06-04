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

data = pd.read_csv('data/train_fnn.csv')
cols = ['date', 'week', 'festival', 'country', 'store', 'product']

trains = data[cols]
labels = data.num_sold


# 构建模型
def build_model():
    model = Sequential([
        Dense(66, activation=tf.nn.relu, input_shape=[6]),
        Dense(6, activation=tf.nn.relu),
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

# 构建模型
#model = build_model()
model = load_model("tabular_fnn_1.h5")
# 当监测数量停止改善时停止训练
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=50)
save_func = callbacks.ModelCheckpoint(filepath='tabular_fnn_2.h5', save_best_only=True)

# 训练模型
H = model.fit(trains, labels, epochs=500, verbose=1, validation_split = 0.1, callbacks=[PrintDot(),save_func])#early_stop,

#model.save("tabular_fnn_2.h5")

# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.show()

