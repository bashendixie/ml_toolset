# import the necessary packages
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import tensorflow as tf


def get_training_model(batchSize, height, width, channel, stnLayer,
                       numClasses, filter):
    # define the input layer and pass the input through the STN
    # layer
    inputs = Input((height, width, channel), batch_size=batchSize)
    x = Lambda(lambda image: tf.cast(image, "float32") / 255.0)(inputs)
    x = stnLayer(x)

    # apply a series of conv and maxpool layers
    x = Conv2D(filter // 4, 3, activation="relu",
               kernel_initializer="he_normal")(x)
    x = MaxPool2D()(x)
    x = Conv2D(filter // 2, 3, activation="relu",
               kernel_initializer="he_normal")(x)
    x = MaxPool2D()(x)
    x = Conv2D(filter, 3, activation="relu",
               kernel_initializer="he_normal")(x)
    x = MaxPool2D()(x)
    # global average pool the output of the previous layer
    x = GlobalAveragePooling2D()(x)
    # pass the flattened output through a couple of dense layers
    x = Dense(filter, activation="relu",
              kernel_initializer="he_normal")(x)
    x = Dense(filter // 2, activation="relu",
              kernel_initializer="he_normal")(x)
    # apply dropout for better regularization
    x = Dropout(0.5)(x)
    # apply softmax to the output for a multi-classification task
    outputs = Dense(numClasses, activation="softmax")(x)
    # return the model
    return Model(inputs, outputs)