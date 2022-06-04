#import tensorflow as tf
#from fcn8s import FCN8s
from utils import DataGenerator
import os
from fcn8s import *
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

TrainSet = DataGenerator("./data/train_image.txt", "./data/train_labels", 2)
#model = FCN8s(n_class=21)

model.compile(optimizer=Adam(lr=1e-4), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
#model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
## train your FCN8s model
callback = ModelCheckpoint("FCN8s.h5", monitor='loss', verbose=1, save_best_only=True)#save_weights_only=True
#model.fit(TrainSet, steps_per_epoch=6000, epochs=30, callbacks=[callback])
model.fit_generator(TrainSet, steps_per_epoch=200, epochs=180, callbacks=[callback])