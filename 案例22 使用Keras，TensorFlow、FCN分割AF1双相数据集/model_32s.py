import sys
import os
import os.path as osp
import cv2 as cv
from PIL import Image # 这里需要的库和加载数据的库是一样的

import numpy as np
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import numpy as np 
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.layers import *
from keras.optimizers import *

# 定义模型
project_name = "fcn_segment"

channels = 1
std_shape = (256, 256, channels) # 输入尺寸, std_shape[0]: img_rows, std_shape[1]: img_cols
                                 # 这个尺寸按你的图像来, 如果你的图大小不一, 那 img_rows 和 image_cols
                                 # 都要设置成 None, 如果你在用 Generator 加载数据时有扩展边缘, 那 std_shape
                                 # 就是扩展后的尺寸

model = keras.Sequential(name = project_name)

model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "relu", padding = "same", input_shape = std_shape, name = "conv_1"))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_1"))
model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_2"))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_2"))
model.add(keras.layers.Conv2D(128, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_3"))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_3"))
model.add(keras.layers.Conv2D(256, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_4"))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_4"))
model.add(keras.layers.Conv2D(512, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_5"))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_5"))
model.add(keras.layers.UpSampling2D(size = (32, 32), interpolation = "nearest", name = "upsamping_6"))

# 这里只有一个卷积核, 可以把 kernel_size 改成 1 * 1, 也可以是其他的, 只是要注意 padding 的尺寸
# 也可以放到 upsamping_6 的前面, 试着改一下尺寸和顺序看一下效果
# 这里只是说明问题, 尺寸和顺序不一定是最好的
model.add(keras.layers.Conv2D(1, kernel_size = (3, 3), activation = "sigmoid", padding = "same", name = "conv_7"))

#model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.compile(optimizer=Adam(lr=1e-4), loss = "binary_crossentropy", metrics = ["accuracy"])
model.summary()


