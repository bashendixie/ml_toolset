# import the necessary packages
import imagenet_alexnet_config as config
import mxnet as mx
import argparse
import json
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-e", "--epoch", type=int, required=True, help="epoch # to load")
args = vars(ap.parse_args())

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())
# construct the testing image iterator
testIter = mx.io.ImageRecordIter(
    path_imgrec=config.TEST_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=config.BATCH_SIZE,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"])

# load the checkpoint from disk
print("[INFO] loading model...")
checkpointsPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
model = mx.model.FeedForward.load(checkpointsPath, args["epoch"])

# compile the model
model = mx.model.FeedForward(
    ctx=[mx.gpu(0)],
    symbol=model.symbol,
    arg_params=model.arg_params,
    aux_params=model.aux_params)

# make predictions on the testing data
print("[INFO] predicting on test data...")
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5)]
(rank1, rank5) = model.score(testIter, eval_metric=metrics)

# display the rank-1 and rank-5 accuracies
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))