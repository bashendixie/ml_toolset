# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

# import the necessary packages
import alexnet_config as config
from customize.tools.imagetoarraypreprocessor import ImageToArrayPreprocessor
from customize.tools.simplepreprocessor import SimplePreprocessor
from customize.tools.patchpreprocessor import PatchPreprocessor
from customize.tools.meanpreprocessor import MeanPreprocessor
from customize.tools.trainingmonitor import TrainingMonitor
from customize.tools.hdf5datasetgenerator import HDF5DatasetGenerator
from alexnet import AlexNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import json

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True,
                         fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

bsize = 8

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, bsize, aug=aug, preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, bsize, preprocessors=[sp, mp, iap], classes=2)

# initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // bsize,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // bsize,
    epochs=75 * 8,
    max_queue_size=bsize * 2,
    callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()
