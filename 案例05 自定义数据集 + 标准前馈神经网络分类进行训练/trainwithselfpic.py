# 标准前馈神经网络,进行图像分类，效果不好
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import gradient_descent_v2
from imutils import paths
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import random
import pickle
import cv2
import os


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
    data = []
    labels = []

    imagePaths = []
    paths = 'D:/Project/DeepLearn/1dataset/custom/raccoon/'
    # grab the images paths and randomly shuffle them
    imagePaths = sorted(list(getFileList(paths, imagePaths)))
    random.seed(42)
    random.shuffle(imagePaths)

    # 浣熊
    for imagePath in imagePaths:
        # load the images, resize the images to be 32x32 pixels (ignoring
        # aspect ratio), flatten the images into 32x32x3=3072 pixel images
        # into a list, and store the images in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)
        # extract the class masks from the images path and update the
        # labels list
        label = 'raccoon'#imagePath.split(os.path.sep)[-2]
        labels.append(label)

    imagePaths = []
    paths = 'D:/Project/DeepLearn/1dataset/custom/fish/'
    # grab the images paths and randomly shuffle them
    imagePaths = sorted(list(getFileList(paths, imagePaths)))
    random.seed(42)
    random.shuffle(imagePaths)

    # 鱼
    for imagePath in imagePaths:
        # load the images, resize the images to be 32x32 pixels (ignoring
        # aspect ratio), flatten the images into 32x32x3=3072 pixel images
        # into a list, and store the images in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)
        # extract the class masks from the images path and update the
        # labels list
        label = 'fish'#imagePath.split(os.path.sep)[-2]
        labels.append(label)

    imagePaths = []
    paths = 'D:/Project/DeepLearn/1dataset/custom/cat/'
    # grab the images paths and randomly shuffle them
    imagePaths = sorted(list(getFileList(paths, imagePaths)))
    random.seed(42)
    random.shuffle(imagePaths)

    # 猫
    for imagePath in imagePaths:
        # load the images, resize the images to be 32x32 pixels (ignoring
        # aspect ratio), flatten the images into 32x32x3=3072 pixel images
        # into a list, and store the images in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)
        # extract the class masks from the images path and update the
        # labels list
        label = 'cat'#imagePath.split(os.path.sep)[-2]
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # convert the labels from integers to vectors (for 2-class, binary
    # classification you should use Keras' to_categorical function
    # instead as the scikit-learn's LabelBinarizer will not return a
    # vector)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    # define the 3072-1024-512-3 architecture using Keras
    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
    model.add(Dense(512, activation="sigmoid"))
    model.add(Dense(len(lb.classes_), activation="softmax"))

    # initialize our initial learning rate and # of epochs to train for
    INIT_LR = 0.01
    EPOCHS = 80
    # compile the model using SGD as our optimizer and categorical
    # cross-entropy loss (you'll want to use binary_crossentropy
    # for 2-class classification)
    print("[INFO] training network...")
    opt = gradient_descent_v2.SGD(lr=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

    # train the neural network
    H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY),epochs=EPOCHS, batch_size=32)
    return model, lb, testX, testY, EPOCHS, H


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
    # save the model and masks binarizer to disk
    print("[INFO] serializing network and masks binarizer...")
    model.save('C:/Users/zyh/Desktop/simple_nn_lb.h5', save_format="h5")
    f = open('C:/Users/zyh/Desktop/simple_nn_lb.pickle', "wb")
    f.write(pickle.dumps(lb))
    f.close()

# 测试模型
def testmodel():
    # load the input images and resize it to the target spatial dimensions
    image = cv2.imread('C:/Users/zyh/Desktop/2.jpg')
    output = image.copy()
    image = cv2.resize(image, (32, 32))
    # scale the pixel values to [0, 1]
    image = image.astype("float") / 255.0
    # check to see if we should flatten the images and add a batch
    # dimension
    if 1 > 0:
        image = image.flatten()
        image = image.reshape((1, image.shape[0]))
    # otherwise, we must be working with a CNN -- don't flatten the
    # images, simply add the batch dimension
    else:
        image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))

    # load the model and masks binarizer
    print("[INFO] loading network and masks binarizer...")
    model = load_model('C:/Users/zyh/Desktop/simple_nn_lb.h5')
    lb = pickle.loads(open('C:/Users/zyh/Desktop/simple_nn_lb.pickle', "rb").read())
    # make a prediction on the images
    preds = model.predict(image)
    # find the class masks index with the largest corresponding
    # probability
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]
    #array([[5.4622066e-01, 4.5377851e-01, 7.7963534e-07]], dtype=float32)
    # draw the class masks + probability on the output images
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)
    # show the output images
    cv2.imshow("Image", output)
    cv2.waitKey(0)

#testmodel()
model, lb, testX, testY, EPOCHS, H = train()
evaluate(model, testX, testY, EPOCHS, H)
savemodel(model, lb)