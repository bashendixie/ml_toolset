import sys
import os
import os.path as osp
import cv2 as cv
from PIL import Image # 这里需要的库和加载数据的库是一样的

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf

from data import *
from model_8s import *
 
 
 # 训练模型
epochs = 1
batch_size = 2	# 当训练图像的尺寸不一样时, 就只能是 1, 不然会把输入数据 shape 不对

# train_path 和 valid_path 由前面的 get_data_path 得到
# 这两个 reader 源源不断地提供训练所需要的数据, 这就是 yield 神奇的地方
# train_reader = segment_reader(train_path, batch_size, shuffle_enable = True)
# valid_reader = segment_reader(valid_path, batch_size, shuffle_enable = True)
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(batch_size,'data/train_af1','images','masks', data_gen_args, save_to_dir = None) # 'rgb', 'rgb',

model_checkpoint = ModelCheckpoint('fcn32_af1_fun.hdf5', monitor='loss', verbose=1, save_best_only=True)

history = model.fit_generator(myGene, 	# 训练数据
                    steps_per_epoch = 1950, #len(train_path) # batch_size,
                    epochs = epochs,
                    #verbose = 1,
                    #validation_data = valid_reader,	# 验证数据
                    #validation_steps = max(1, len(valid_path) # batch_size),
                    #max_queue_size = 8,
                    #workers = 1,
                    callbacks=[model_checkpoint])

                    
tf.saved_model.save(model, "save_test")
#model = tf.saved_model.load("save_test")