# import the necessary packages
import age_gender_config as config
from mxagegendernet import MxAgeGenderNet
from agegenderhelper import AgeGenderHelper
from mxmetrics import one_off_callback
import mxnet as mx
import argparse
import logging
import pickle
import json
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=False, help="path to output checkpoint directory", default="checkpoints")
ap.add_argument("-p", "--prefix", required=False, help="name of model prefix", default="mxagegendernet")
ap.add_argument("-s", "--start-epoch", type=int, default=0,help="epoch to restart training at")
args = vars(ap.parse_args())

# set the logging level and output file
logging.basicConfig(level=logging.DEBUG, filename="training_{}.log".format(args["start_epoch"]), filemode="w")

# determine the batch and load the mean pixel values
batchSize = config.BATCH_SIZE * config.NUM_DEVICES
means = json.loads(open(config.DATASET_MEAN).read())


# initialize the optimizer
opt = mx.optimizer.SGD(learning_rate=1e-3, momentum=0.9, wd=0.0005, rescale_grad=1.0 / batchSize)

# construct the training image iterator
trainIter = mx.io.ImageRecordIter(
    path_imgrec=config.TRAIN_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=batchSize,
    rand_crop=True,
    rand_mirror=True,
    rotate=7,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"],
    preprocess_threads=config.NUM_DEVICES * 2)

# construct the validation image iterator
valIter = mx.io.ImageRecordIter(
    path_imgrec=config.VAL_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=batchSize,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"])


# construct the checkpoints path, initialize the model argument and
# auxiliary parameters
checkpointsPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
argParams = None
auxParams = None

# if there is no specific model starting epoch supplied, then
# initialize the network
if args["start_epoch"] <= 0:
    # build the LeNet architecture
    print("[INFO] building network...")
    model = MxAgeGenderNet.build(config.NUM_CLASSES)

# otherwise, a specific checkpoint was supplied
else:
    # load the checkpoint from disk
    print("[INFO] loading epoch {}...".format(args["start_epoch"]))
    (model, argParams, auxParams) = mx.model.load_checkpoint(checkpointsPath, args["start_epoch"])

# compile the model
model = mx.model.FeedForward(
    #ctx=[mx.gpu(2), mx.gpu(3)],
    symbol=model,
    initializer=mx.initializer.Xavier(),
    arg_params=argParams,
    aux_params=auxParams,
    optimizer=opt,
    num_epoch=110,
    begin_epoch=args["start_epoch"])

# initialize the callbacks and evaluation metrics
batchEndCBs = [mx.callback.Speedometer(batchSize, 10)]
epochEndCBs = [mx.callback.do_checkpoint(checkpointsPath)]
metrics = [mx.metric.Accuracy(), mx.metric.CrossEntropy()]

# check to see if the one-off accuracy callback should be used
if config.DATASET_MEAN == "age":
    # load the label encoder, then build the one-off mappings for
    # computing accuracy
    le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
    agh = AgeGenderHelper(config)
    oneOff = agh.buildOneOffMappings(le)
    epochEndCBs.append(one_off_callback(trainIter, valIter, oneOff, mx.gpu(0)))

# train the network
print("[INFO] training network...")
model.fit(
    X=trainIter,
    eval_data=valIter,
    eval_metric=metrics,
    batch_end_callback=batchEndCBs,
    epoch_end_callback=epochEndCBs)
