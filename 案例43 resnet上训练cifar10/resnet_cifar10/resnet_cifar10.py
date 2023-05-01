# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from models.resnet.resnet import ResNet
#允许我们从特定检查点停止和重新开始训练
from customize.tools.epochcheckpoint import EpochCheckpoint
from customize.tools.trainingmonitor import TrainingMonitor
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np
import argparse
import sys

# set a high recursion limit so Theano doesn’t complain
# 如果使用Theano作为后端，如果使用tensorflow则不用管这行
sys.setrecursionlimit(5000)

# construct the argument parse and parse the arguments
# 我们的脚本只需要 --checkpoints 开关，这是我们每 N 个时期存储 ResNet 权重的目录的路径。
# 如果我们需要从特定时期重新开始训练，我们可以提供 --model 路径以及指示特定时期编号的整数。
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

# load the training and testing data, converting the images from integers to floats
# 下一步是从磁盘加载 CIFAR-10 数据集（预拆分为训练和测试），执行均值减法，并将整数标签单热编码为向量
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the images generator for data augmentation
# 初始化一个 ImageDataGenerator，以便我们可以将数据增强应用于 CIFAR-10
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

# if there is no specific model checkpoint supplied, then initialize
# the network (ResNet-56) and compile the model
# 如果我们从第一个 epoch 开始训练 ResNet，我们需要实例化网络架构
if args["model"] is None:
    print("[INFO] compiling model...")
    opt = SGD(lr=1e-1)
    model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # otherwise, load the checkpoint from disk
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)
    print("[INFO] new learning rate: {}".format( K.get_value(model.optimizer.lr)))

# construct the set of callbacks
callbacks = [EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
TrainingMonitor("output/resnet56_cifar10.png", jsonPath="output/resnet56_cifar10.json", startAt=args["start_epoch"])]

# train the network
print("[INFO] training network...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=128), validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // 128, epochs=10,
    callbacks=callbacks, verbose=1)