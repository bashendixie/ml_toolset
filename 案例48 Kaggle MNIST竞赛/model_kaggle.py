# import the necessary packages
import tensorflow.keras.models
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

model = tensorflow.keras.models.load_model("lenet_mnist.v9.h5")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y_train = train['masks']
x_train = train.drop('masks', 1)

x_train = x_train.values.reshape((x_train.shape[0], 28, 28, 1))
x_train = x_train.astype("float32") / 255.0

x_test = test.values.reshape((test.shape[0], 28, 28, 1))
x_test = x_test.astype("float32") / 255.0


### 预测测试数据并生成可提交的文件
x_train_predictions = model.predict(x_test, batch_size=1000)
y_res = []
for item in x_train_predictions:
    item = item.tolist()
    y_res.append(item.index(max(item)))

conv_submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': np.int32(y_res)})
conv_submission.to_csv('predict_v2.csv', index=False)

### 预测训练数据,用以测试模型
# r = 0
# e = 0
# i = 0
# print("预测开始")
# x_train_predictions = model.predict(x_train, batch_size=1000)
# for item in x_train_predictions:
#     item = item.tolist()
#     if y_train[i] == item.index(max(item)):
#         r = r + 1
#     else:
#         e = e + 1
#     i = i+1
#
# print('正确数量= %d' %(r))
# print('错误数量= %d' %(e))
# print("预测完成")

