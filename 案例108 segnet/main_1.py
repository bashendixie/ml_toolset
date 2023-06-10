import tensorflow as tf
from tensorflow.python.keras.models import *
from keras.layers import *
from keras.layers.normalization.layer_normalization import *
from keras.layers.normalization.batch_normalization import *
from keras.optimizers import *
import keras.losses as losses
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import os


tf.random.set_seed(42)
np.random.seed(42)

train_size = 0.8
lr = 3e-4
weight_decay = 8e-9
batch_size = 16
epochs = 1


def load_dataset(train_part, root='PH2Dataset'):
    images = []
    masks = []

    for root, dirs, files in os.walk(os.path.join(root, 'PH2 Dataset images')):
        if root.endswith('_Dermoscopic_Image'):
            images.append(imread(os.path.join(root, files[0])))
        if root.endswith('_lesion'):
            masks.append(imread(os.path.join(root, files[0])))

    size = (256, 256)
    images = np.array([resize(image, size, mode='constant', anti_aliasing=True, ) for image in images])
    masks = np.expand_dims(np.array([resize(mask, size, mode='constant', anti_aliasing=False) > 0.5 for mask in masks]),
                           axis=3)

    indices = np.random.permutation(range(len(images)))
    train_part = int(train_part * len(images))
    train_ind = indices[:train_part]
    test_ind = indices[train_part:]

    X_train = tf.cast(images[train_ind, :, :, :], tf.float32)
    y_train = tf.cast(masks[train_ind, :, :, :], tf.float32)

    X_test = tf.cast(images[test_ind, :, :, :], tf.float32)
    y_test = tf.cast(masks[test_ind, :, :, :], tf.float32)

    return (X_train, y_train), (X_test, y_test)


def plotn(n, data):
    images, masks = data[0], data[1]
    fig, ax = plt.subplots(1, n)
    fig1, ax1 = plt.subplots(1, n)
    for i, (img, mask) in enumerate(zip(images, masks)):
        if i == n:
            break
        ax[i].imshow(img)
        ax1[i].imshow(mask[:, :, 0])
    plt.show()


class SegNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.enc_conv0 = Conv2D(16, kernel_size=3, padding='same')
        self.bn0 = BatchNormalization()
        self.relu0 = Activation('relu')
        self.pool0 = MaxPool2D()

        self.enc_conv1 = Conv2D(32, kernel_size=3, padding='same')
        self.relu1 = Activation('relu')
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPool2D()

        self.enc_conv2 = Conv2D(64, kernel_size=3, padding='same')
        self.relu2 = Activation('relu')
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPool2D()

        self.enc_conv3 = Conv2D(128, kernel_size=3, padding='same')
        self.relu3 = Activation('relu')
        self.bn3 = BatchNormalization()
        self.pool3 = MaxPool2D()

        self.bottleneck_conv = Conv2D(256, kernel_size=(3, 3), padding='same')

        self.upsample0 = UpSampling2D(interpolation='bilinear')
        self.dec_conv0 = Conv2D(128, kernel_size=3, padding='same')
        self.dec_relu0 = Activation('relu')
        self.dec_bn0 = BatchNormalization()

        self.upsample1 = UpSampling2D(interpolation='bilinear')
        self.dec_conv1 = Conv2D(64, kernel_size=3, padding='same')
        self.dec_relu1 = Activation('relu')
        self.dec_bn1 = BatchNormalization()

        self.upsample2 = UpSampling2D(interpolation='bilinear')
        self.dec_conv2 = Conv2D(32, kernel_size=3, padding='same')
        self.dec_relu2 = Activation('relu')
        self.dec_bn2 = BatchNormalization()

        self.upsample3 = UpSampling2D(interpolation='bilinear')
        self.dec_conv3 = Conv2D(1, kernel_size=1)

    def call(self, input):
        e0 = self.pool0(self.relu0(self.bn0(self.enc_conv0(input))))
        e1 = self.pool1(self.relu1(self.bn1(self.enc_conv1(e0))))
        e2 = self.pool2(self.relu2(self.bn2(self.enc_conv2(e1))))
        e3 = self.pool3(self.relu3(self.bn3(self.enc_conv3(e2))))

        b = self.bottleneck_conv(e3)

        d0 = self.dec_relu0(self.dec_bn0(self.upsample0(self.dec_conv0(b))))
        d1 = self.dec_relu1(self.dec_bn1(self.upsample1(self.dec_conv1(d0))))
        d2 = self.dec_relu2(self.dec_bn2(self.upsample2(self.dec_conv2(d1))))
        d3 = self.dec_conv3(self.upsample3(d2))

        return d3


def train(datasets, model, epochs, batch_size):
    train_dataset, test_dataset = datasets[0], datasets[1]
    model.fit(train_dataset[0], train_dataset[1],
              epochs=epochs, batch_size=batch_size, shuffle=True,
              validation_data=(test_dataset[0], test_dataset[1]))
    model.save('C:/Users/zyh/Desktop/segnet', save_format="tf")


# 加载数据集
(X_train, y_train), (X_test, y_test) = load_dataset(train_size)
# 声明模型
model = SegNet()
# 优化器
opt = adam_v2.Adam(learning_rate=lr, decay=weight_decay)
# 定义损失
loss_fn = losses.BinaryCrossentropy(from_logits=True)
# 编译
model.compile(loss=loss_fn, optimizer=opt)
# 进行训练
train(((X_train, y_train), (X_test, y_test)), model, epochs, batch_size)

# 进行预测
predictions = []
image_mask = []
plots = 5

for i, (img, mask) in enumerate(zip(X_test, y_test)):
    if i == plots:
        break
    img = tf.expand_dims(img, 0)
    pred = np.array(model.predict(img))
    predictions.append(pred[0, :, :, 0] > 0.5)
    image_mask.append(mask)
plotn(plots, (predictions, image_mask))