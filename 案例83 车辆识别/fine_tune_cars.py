# import the necessary packages
import mxnet as mx
import car_config as config
import argparse
import logging
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--vgg", required=False, help="path to pre-trained VGGNet for fine-tuning", default="D:/Project/deeplearn/预训练模型/vgg16_zoo/vgg16")
ap.add_argument("-c", "--checkpoints", required=False, help="path to output checkpoint directory", default="checkpoints")
ap.add_argument("-p", "--prefix", required=False, help="name of model prefix", default="vggnet")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

# set the logging level and output file
logging.basicConfig(level=logging.DEBUG, filename="training_{}.log".format(args["start_epoch"]), filemode="w")

# determine the batch
batchSize = config.BATCH_SIZE * config.NUM_DEVICES

# construct the training image iterator
trainIter = mx.io.ImageRecordIter(
    path_imgrec=config.TRAIN_MX_REC,
    data_shape=(3, 224, 224),
    batch_size=batchSize,
    rand_crop=True,
    rand_mirror=True,
    rotate=15,
    max_shear_ratio=0.1,
    mean_r=config.R_MEAN,
    mean_g=config.G_MEAN,
    mean_b=config.B_MEAN,
    preprocess_threads=config.NUM_DEVICES * 2)

# construct the validation image iterator
valIter = mx.io.ImageRecordIter(
    path_imgrec=config.VAL_MX_REC,
    data_shape=(3, 224, 224),
    batch_size=batchSize,
    mean_r=config.R_MEAN,
    mean_g=config.G_MEAN,
    mean_b=config.B_MEAN)

# initialize the optimizer and the training contexts
opt = mx.optimizer.SGD(learning_rate=1e-4, momentum=0.9, wd=0.0005, rescale_grad=1.0 / batchSize)
ctx = [mx.gpu(3)]

# construct the checkpoints path, initialize the model argument and
# auxiliary parameters, and whether uninitialized parameters should
# be allowed
checkpointsPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
argParams = None
auxParams = None
allowMissing = False

# if there is no specific model starting epoch supplied, then we
# need to build the network architecture
if args["start_epoch"] <= 0:
    # load the pre-trained VGG16 model
    print("[INFO] loading pre-trained model...")
    (symbol, argParams, auxParams) = mx.model.load_checkpoint(args["vgg"], 0)
    allowMissing = True

    # grab the layers from the pre-trained model, then find the
    # dropout layer *prior* to the final FC layer (i.e., the layer
    # that contains the number of class labels)
    # HINT: you can find layer names like this:
    # for layer in layers:
    # print(layer.name)
    # then, append the string ‘_output‘ to the layer name
    layers = symbol.get_internals()
    net = layers["drop7_output"]

    # construct a new FC layer using the desired number of output
    # class labels, followed by a softmax output
    net = mx.sym.FullyConnected(data=net, num_hidden = config.NUM_CLASSES, name = "fc8")
    net = mx.sym.SoftmaxOutput(data=net, name="softmax")
    # construct a new set of network arguments, removing any previous
    # arguments pertaining to FC8 (this will allow us to train the
    # final layer)
    argParams = dict({k: argParams[k] for k in argParams if "fc8" not in k})

# otherwise, a specific checkpoint was supplied
else:
    # load the checkpoint from disk
    print("[INFO] loading epoch {}...".format(args["start_epoch"]))
    (net, argParams, auxParams) = mx.model.load_checkpoint(checkpointsPath, args["start_epoch"])

# initialize the callbacks and evaluation metrics
batchEndCBs = [mx.callback.Speedometer(batchSize, 50)]
epochEndCBs = [mx.callback.do_checkpoint(checkpointsPath)]
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5), mx.metric.CrossEntropy()]

# construct the model and train it
print("[INFO] training network...")
model = mx.mod.Module(symbol=net, context=ctx)
model.fit(
    trainIter,
    eval_data=valIter,
    num_epoch=65,
    begin_epoch=args["start_epoch"],
    initializer=mx.initializer.Xavier(),
    arg_params=argParams,
    aux_params=auxParams,
    optimizer=opt,
    allow_missing=allowMissing,
    eval_metric=metrics,
    batch_end_callback=batchEndCBs,
    epoch_end_callback=epochEndCBs)




