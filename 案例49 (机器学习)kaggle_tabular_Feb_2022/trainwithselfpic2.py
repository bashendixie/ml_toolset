# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical, plot_model

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from imutils import paths
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.optimizers as optimizers
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import matplotlib
matplotlib.use("Agg")


class SmallVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 3 => POOL layer set
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model


def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist

# train的方法
def train():
    # initialize the data and labels
    print("[INFO] loading images...")
    paths = 'to_img/train/'
    # initialize our VGG-like Convolutional Neural Network
    model = load_model('tabular_vgg_0.9371_1.h5')
    #model = SmallVGGNet.build(width=286, height=286, depth=1, classes=10)
    EPOCHS = 1000
    batch_size = 2
    train_image_generator = ImageDataGenerator(rescale=1/255)
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=paths,
                                                               color_mode='grayscale',
                                                               shuffle=True,
                                                               target_size=(286, 286),
                                                               class_mode='categorical')

    print("[INFO] training network...")
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=1e-3), metrics=["accuracy"])
    # train the network
    # 当监测数量停止改善时停止训练
    early_stop = callbacks.EarlyStopping(monitor='loss', patience=50)
    save_func = callbacks.ModelCheckpoint(filepath='tabular_vgg_2.h5', monitor='loss', save_best_only=True)
    H = model.fit(train_data_gen, epochs=EPOCHS, verbose=1, callbacks=[early_stop, save_func])#steps_per_epoch=10000,
    return model, lb, EPOCHS, H


# 评估的方法,绘制训练损失和准确性
def evaluate(model, testX, testY, EPOCHS, H):
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(x=testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=lb.classes_))
    # plot the training loss and accuracy
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["accuracy"], label="train_acc")
    plt.plot(N, H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('C:/Users/zyh/Desktop/simple_nn_plot.png')

# 保存模型
def savemodel(model, lb):
    # save the model and label binarizer to disk
    print("[INFO] serializing network and label binarizer...")
    model.save('C:/Users/zyh/Desktop/simple_nn_lb.h5', save_format="h5")
    f = open('C:/Users/zyh/Desktop/simple_nn_lb.pickle', "wb")
    f.write(pickle.dumps(lb))
    f.close()

# 测试模型
def testmodel():
    # load the input image and resize it to the target spatial dimensions
    image = cv2.imread('C:/Users/zyh/Desktop/2.jpg')
    output = image.copy()
    image = cv2.resize(image, (64, 64))
    # scale the pixel values to [0, 1]
    image = image.astype("float") / 255.0
    # check to see if we should flatten the image and add a batch
    # dimension
    if -1 > 0:
        image = image.flatten()
        image = image.reshape((1, image.shape[0]))
    # otherwise, we must be working with a CNN -- don't flatten the
    # image, simply add the batch dimension
    else:
        image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))

    # load the model and label binarizer
    print("[INFO] loading network and label binarizer...")
    model = load_model('C:/Users/zyh/Desktop/tabular_vgg_0.9371_1.h5')
    lb = pickle.loads(open('C:/Users/zyh/Desktop/simple_nn_lb.pickle', "rb").read())
    # make a prediction on the image
    preds = model.predict(image)
    # find the class label index with the largest corresponding
    # probability
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]
    #array([[5.4622066e-01, 4.5377851e-01, 7.7963534e-07]], dtype=float32)
    # draw the class label + probability on the output image
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)
    # show the output image
    cv2.imshow("Image", output)
    cv2.waitKey(0)


#testmodel()
model, lb, testX, testY, EPOCHS, H = train()
evaluate(model, testX, testY, EPOCHS, H)
