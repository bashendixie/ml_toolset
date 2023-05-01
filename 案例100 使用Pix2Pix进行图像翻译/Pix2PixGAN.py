# import the necessary packages
from keras.layers import BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import concatenate
from keras.layers import MaxPool2D
from keras.layers import Conv2D
from keras.layers import Dropout
from keras import Model
from keras import Input


class Pix2Pix(object):
    def __init__(self, imageHeight, imageWidth):
        # initialize the images height and width
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth

    def generator(self):
        # initialize the input layer
        inputs = Input([self.imageHeight, self.imageWidth, 3])

        # down Layer 1 (d1) => final layer 1 (f1)
        d1 = Conv2D(32, (3, 3), activation="relu", padding="same")(
            inputs)
        d1 = Dropout(0.1)(d1)
        f1 = MaxPool2D((2, 2))(d1)

        # down Layer 2 (l2) => final layer 2 (f2)
        d2 = Conv2D(64, (3, 3), activation="relu", padding="same")(f1)
        f2 = MaxPool2D((2, 2))(d2)

        #  down Layer 3 (l3) => final layer 3 (f3)
        d3 = Conv2D(96, (3, 3), activation="relu", padding="same")(f2)
        f3 = MaxPool2D((2, 2))(d3)

        # down Layer 4 (l3) => final layer 4 (f4)
        d4 = Conv2D(96, (3, 3), activation="relu", padding="same")(f3)
        f4 = MaxPool2D((2, 2))(d4)

        # u-bend of the u-bet
        b5 = Conv2D(96, (3, 3), activation="relu", padding="same")(f4)
        b5 = Dropout(0.3)(b5)
        b5 = Conv2D(256, (3, 3), activation="relu", padding="same")(b5)

        # upsample Layer 6 (u6)
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2),
                             padding="same")(b5)
        u6 = concatenate([u6, d4])
        u6 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            u6)

        # upsample Layer 7 (u7)
        u7 = Conv2DTranspose(96, (2, 2), strides=(2, 2),
                             padding="same")(u6)
        u7 = concatenate([u7, d3])
        u7 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            u7)

        # upsample Layer 8 (u8)
        u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2),
                             padding="same")(u7)
        u8 = concatenate([u8, d2])
        u8 = Conv2D(128, (3, 3), activation="relu", padding="same")(u8)

        # upsample Layer 9 (u9)
        u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2),
                             padding="same")(u8)
        u9 = concatenate([u9, d1])
        u9 = Dropout(0.1)(u9)
        u9 = Conv2D(128, (3, 3), activation="relu", padding="same")(u9)

        # final conv2D layer
        outputLayer = Conv2D(3, (1, 1), activation="tanh")(u9)

        # create the generator model
        generator = Model(inputs, outputLayer)

        # return the generator
        return generator

    def discriminator(self):
        # initialize input layer according to PatchGAN
        inputMask = Input(shape=[self.imageHeight, self.imageWidth, 3],
                          name="input_image"
                          )
        targetImage = Input(
            shape=[self.imageHeight, self.imageWidth, 3],
            name="target_image"
        )

        # concatenate the inputs
        x = concatenate([inputMask, targetImage])

        # add four conv2D convolution layers
        x = Conv2D(64, 4, strides=2, padding="same")(x)
        x = LeakyReLU()(x)
        x = Conv2D(128, 4, strides=2, padding="same")(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, 4, strides=2, padding="same")(x)
        x = LeakyReLU()(x)
        x = Conv2D(512, 4, strides=1, padding="same")(x)

        # add a batch-normalization layer => LeakyReLU => zeropad
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # final conv layer
        last = Conv2D(1, 3, strides=1)(x)

        # create the discriminator model
        discriminator = Model(inputs=[inputMask, targetImage],
                              outputs=last)

        # return the discriminator
        return discriminator