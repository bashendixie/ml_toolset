# USAGE
# python train.py
# setting seed for reproducibility
import tensorflow as tf
tf.random.set_seed(42)
# import the necessary packages
from spatial_transformer import STN
from model import get_training_model
from callback import get_train_monitor
import config
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_datasets as tfds
import os
# load the train and test dataset
print("[INFO] loading the train and test dataset...")
trainingDs = tfds.load(name=config.DATASET_NAME,
	data_dir=config.DATASET_PATH, split="train", shuffle_files=True,
	as_supervised=True)
testingDs = tfds.load(name=config.DATASET_NAME,
	data_dir=config.DATASET_PATH, split="test", as_supervised=True)
 # preprocess the train and test dataset
print("[INFO] preprocessing the train and test dataset...")
trainDs = (
	trainingDs
	.shuffle(config.BATCH_SIZE*100)
	.batch(config.BATCH_SIZE, drop_remainder=True)
	.prefetch(config.AUTO)
)
testDs = (
	testingDs
	.batch(config.BATCH_SIZE, drop_remainder=True)
	.prefetch(config.AUTO)
)
# initialize the stn layer
print("[INFO] initializing the stn layer...")
stnLayer = STN(name=config.STN_LAYER_NAME, filter=config.FILTERS)
# get the classification model for cifar10
print("[INFO] grabbing the multiclass classification model...")
model = get_training_model(batchSize=config.BATCH_SIZE,
	height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH,
	channel=config.CHANNEL, stnLayer=stnLayer,
	numClasses=config.CLASSES, filter=config.FILTERS)
# print the model summary
print("[INFO] the model summary...")
print(model.summary())
# create an output images directory if it not already exists
if not os.path.exists(config.OUTPUT_PATH):
	os.makedirs(config.OUTPUT_PATH)
# get the training monitor
trainMonitor = get_train_monitor(testDs=testDs,
	outputPath=config.OUTPUT_PATH, stnLayerName=config.STN_LAYER_NAME)
# compile the model
print("[INFO] compiling the model...")
model.compile(loss=config.LOSS_FN, optimizer=config.OPTIMIZER,
	metrics=["accuracy"])
# define an early stopping callback
esCallback = EarlyStopping(patience=5, restore_best_weights=True)
# train the model
print("[INFO] training the model...")
model.fit(trainDs, epochs=config.EPOCHS,
	callbacks=[trainMonitor, esCallback], validation_data=testDs)