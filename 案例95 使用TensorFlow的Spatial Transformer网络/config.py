# import the necessary packages
from tensorflow.data import AUTOTUNE

import os
# define AUTOTUNE
AUTO = AUTOTUNE
# define the image height, width and channel size
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
CHANNEL = 1
# define the dataset path, dataset name, and the batch size
DATASET_PATH = "dataset"
DATASET_NAME = "emnist"
BATCH_SIZE = 1024
# define the number of epochs
EPOCHS = 100
# define the conv filters
FILTERS = 256
# define an output directory
OUTPUT_PATH = "output"
# define the loss function and optimizer
LOSS_FN = "sparse_categorical_crossentropy"
OPTIMIZER = "adam"
# define the name of the gif
GIF_NAME = "stn.gif"
# define the number of classes for classification
CLASSES = 62
# define the stn layer name
STN_LAYER_NAME = "stn"