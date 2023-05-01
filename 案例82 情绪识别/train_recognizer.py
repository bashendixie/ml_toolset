# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import emotion_config as config
from customize.tools.imagetoarraypreprocessor import ImageToArrayPreprocessor
from customize.tools.epochcheckpoint import EpochCheckpoint
from customize.tools.trainingmonitor import TrainingMonitor
from customize.tools.hdf5datasetgenerator import HDF5DatasetGenerator
from emotionvggnet import EmotionVGGNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=False, help="path to output checkpoint directory", default=config.CHECKPOINTS_PATH)
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")#, default=config.MODEL_PATH
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

# construct the training and testing images generators for data
# augmentation, then initialize the images preprocessor
trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True, rescale=1 / 255.0, fill_mode="nearest")
valAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=trainAug, preprocessors=[iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=valAug, preprocessors=[iap], classes=config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model"] is None:
    print("[INFO] compiling model...")
    model = EmotionVGGNet.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
    opt = Adam(lr=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# otherwise, load the checkpoint from disk
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-3)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

# construct the set of callbacks
figPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.png"])
jsonPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.json"])
callbacks = [
    EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
    TrainingMonitor(figPath, jsonPath=jsonPath, startAt=args["start_epoch"])]


# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // config.BATCH_SIZE,
    epochs=15,
    max_queue_size=config.BATCH_SIZE * 2,
    callbacks=callbacks, verbose=1)

# close the databases
trainGen.close()
valGen.close()








