# 孪生神经网络（完整代码）
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from keras.datasets import mnist
import numpy as np
import cv2
import os
import tensorflow.python.keras.backend as k
import matplotlib.pyplot as plt
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

# 参数设置
# specify the shape of the inputs for our network
IMG_SHAPE = (28, 28, 1)
# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 100

# define the path to the base output directory
BASE_OUTPUT = "C:/Users/zyh/Desktop"
# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])


# 创建孪生网络模型的方法
def build_siamese_model(input_shape, embedding_dim=48):
    # specify the inputs for the feature extractor network
    inputs = Input(input_shape)
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    # prepare the final outputs
    pooled_output = GlobalAveragePooling2D()(x)
    out_puts = Dense(embedding_dim)(pooled_output)
    # build the model
    my_model = Model(inputs, out_puts)
    # return the model to the calling function
    return my_model


# 创建图像对的方法
def make_pairs(images, labels):
    # 初始化两个空数组（图像对）和（标签，用来说明图像对是正向还是负向）
    pair_images = []
    pair_labels = []
    # 计算数据集中存在的类总数，然后为每个类标签建立索引列表，该列表为具有给定标签的所有示例提供索引
    num_classes = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, num_classes)]

    # 遍历所有图像
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        current_image = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idx_b = np.random.choice(idx[label])
        pos_image = images[idx_b]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pair_images.append([current_image, pos_image])
        pair_labels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        neg_idx = np.where(labels != label)[0]
        neg_image = images[np.random.choice(neg_idx)]
        # prepare a negative pair of images and update our lists
        pair_images.append([current_image, neg_image])
        pair_labels.append([0])
        # return a 2-tuple of our image pairs and labels

    return np.array(pair_images), np.array(pair_labels)


# 保存训练数据
def plot_training(h, plot_path):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(h.history["loss"], label="train_loss")
    plt.plot(h.history["val_loss"], label="val_loss")
    plt.plot(h.history["accuracy"], label="train_acc")
    plt.plot(h.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot_path)


# 计算欧氏距离
def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsAA, featsBB) = vectors
    # compute the sum of squared distances between the vectors
    sum_squared = k.sum(k.square(featsAA - featsBB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return k.sqrt(k.maximum(sum_squared, k.epsilon()))


# 加载数据集并且归一化
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data('C:/Users/zyh/Desktop/mnist.npz')
trainX = trainX / 255.0
testX = testX / 255.0
# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = make_pairs(trainX, trainY)
(pairTest, labelTest) = make_pairs(testX, testY)


# 配置孪生神经网络
print("[INFO] building siamese network...")
imgA = Input(IMG_SHAPE)
imgB = Input(IMG_SHAPE)
featureExtractor = build_siamese_model(IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# 建立神经网络
distance = euclidean_distance([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

# compile the model
print("[INFO] compiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam",	metrics=["accuracy"])
# train the model
print("[INFO] training model...")
history = model.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
                    validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS)

# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(MODEL_PATH)
# plot the training history
print("[INFO] plotting training history...")
plot_training(history, PLOT_PATH)
