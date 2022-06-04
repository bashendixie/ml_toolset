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

img_input = keras.layers.Input(shape = std_shape, name = "input")

conv_1 = keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_1")(img_input)
max_pool_1 = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_1")(conv_1)

conv_2 = keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_2")(max_pool_1)
max_pool_2 = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_2")(conv_2)

conv_3 = keras.layers.Conv2D(128, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_3")(max_pool_2)
max_pool_3 = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_3")(conv_3)

conv_4 = keras.layers.Conv2D(256, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_4")(max_pool_3)
max_pool_4 = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_4")(conv_4)

conv_5 = keras.layers.Conv2D(512, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_5")(max_pool_4)
max_pool_5 = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_5")(conv_5)

# max_pool_5 转置卷积上采样 2 倍至 max_pool_4 一样大
up6 = keras.layers.Conv2DTranspose(256, kernel_size = (3, 3), strides = (2, 2), padding = "same", kernel_initializer = "he_normal", name = "upsamping_6")(max_pool_5)
                
_16s = keras.layers.add([max_pool_4, up6])

# _16s 上采样 16 倍后与输入尺寸相同
up7 = keras.layers.UpSampling2D(size = (16, 16), interpolation = "bilinear", name = "upsamping_7")(_16s)

# 这里 kernel 也是 3 * 3, 也可以同 FCN-32s 那样修改的
conv_7 = keras.layers.Conv2D(1, kernel_size = (3, 3), activation = "sigmoid", padding = "same", name = "conv_7")(up7)

model = keras.Model(img_input, conv_7, name = project_name)

#model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.compile(optimizer=Adam(lr=1e-4), loss = "binary_crossentropy", metrics = ["accuracy"])
model.summary()