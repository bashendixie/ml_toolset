import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import *
from tensorflow import keras
from tensorflow.keras import Sequential
import tensorflow as tf


train_datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1.0/255.0,
    preprocessing_function=None,
    data_format=None,
    dtype=None)
test_datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1.0/255.0,
    preprocessing_function=None,
    data_format=None,
    dtype=None)
train_generator = train_datagen.flow_from_directory("C:/Users/zyh/Desktop/archive/emirhan_human_dataset/datasets/human_data/train_data",target_size=(128, 128),
                                                    batch_size=4,
                                                    class_mode='categorical',
                                                    interpolation="nearest")
test_generator = test_datagen.flow_from_directory("C:/Users/zyh/Desktop/archive/emirhan_human_dataset/datasets/human_data/test_data",target_size=(128, 128),
                                                    batch_size=4,
                                                    class_mode='categorical',
                                                    interpolation="nearest")






# 预训练模型
def func(pre, name_model):
    print('#####~Model => {} '.format(name_model))
    pre_model = name_model(input_shape=(128, 128, 3),
                           include_top=False,
                           weights='imagenet',
                           pooling='avg')
    pre_model.trainable = False
    inputs = pre_model.input
    x = Dense(64, activation='relu')(pre_model.output)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(15, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    my_callbacks = [EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=5,
                                  mode='auto')]

    history = model.fit(train_generator, validation_data=test_generator, epochs=5, callbacks=my_callbacks, verbose=0)
    # Plotting Accuracy, val_accuracy, loss, val_loss
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()

    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['Train', 'Validation'])
    plt.show()

    # Predict Data Test
    pred = model.predict(test_generator)
    pred = np.argmax(pred, axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    pred = [labels[k] for k in pred]

    print('\033[01m              Classification_report \033[0m')

    print('\033[01m              Results \033[0m')
    # Results
    results = model.evaluate(test_generator, verbose=0)
    print("    Test Loss:\033[31m \033[01m {:.5f} \033[30m \033[0m".format(results[0]))
    print("Test Accuracy:\033[32m \033[01m {:.2f}% \033[30m \033[0m".format(results[1] * 100))

    return results

# 自定义模型
def emir_model():
    inp = Input(shape = (128,128,3))

    x = Conv2D(32, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True)(inp)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same', data_format=None)(x)
    x = Conv2D(64, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same', data_format=None)(x)
    x = Conv2D(128, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same', data_format=None)(x)
    x = Conv2D(256, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same', data_format=None)(x)
    x = Conv2D(512, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same', data_format=None)(x)
    x = Conv2D(1024, (3,3), strides=(2, 2), padding='same', activation='relu', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same', data_format=None)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(15, activation='softmax')(x)

    model = Model(inputs=inp, outputs= x)
    return model

# 自定义方法
def emirhan_func(name_model):
    print('#####~Model => {} '.format(name_model))

    model = emir_model()
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    my_callbacks = [EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=5,
                                  mode='auto')]

    history = model.fit(train_generator,
                        validation_data=test_generator,
                        epochs=64,
                        callbacks=my_callbacks,
                        verbose=1,
                        batch_size=4, )
    # Plotting Accuracy, val_accuracy, loss, val_loss
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()

    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['Train', 'Validation'])
    plt.show()

    # Predict Data Test
    pred = model.predict(test_generator)
    pred = np.argmax(pred, axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    pred = [labels[k] for k in pred]

    print('\033[01m              Classification_report \033[0m')

    print('\033[01m              Results \033[0m')
    # Results
    results = model.evaluate(test_generator, verbose=0)
    print("    Test Loss:\033[31m \033[01m {:.5f} \033[30m \033[0m".format(results[0]))
    print("Test Accuracy:\033[32m \033[01m {:.2f}% \033[30m \033[0m".format(results[1] * 100))

    return results





def depth(x, strides):
    x = DepthwiseConv2D(3,strides=strides,padding='same',  use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x
def single_conv_block(x,filters):
    x = Conv2D(filters,1,use_bias=False)(x)
    x= BatchNormalization()(x)
    x = ReLU()(x)
    x = SpatialDropout2D(0.2)(x)
    return x
def hidden_layers(x,filters,strides):
    x = depth(x,strides)
    x = single_conv_block(x, filters)
    return x
def EmirhanModel(input_shape=(128,128,3),n_classes = 15):
    input = Input (input_shape)
    x = Conv2D(32,3,strides=(2,2),padding = 'same', use_bias=False) (input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ZeroPadding2D(padding=(2, 2))(x)
    x = hidden_layers(x,64, strides=(1,1))
    x = hidden_layers(x,128,strides=(2,2))
    x = hidden_layers(x,128,strides=(1,1))
    x = hidden_layers(x,256,strides=(2,2))
    x = hidden_layers(x,256,strides=(1,1))
    x = hidden_layers(x,512,strides=(2,2))
    for _ in range(5):
      x = hidden_layers(x,512,strides=(1,1))
      x = hidden_layers(x,1024,strides=(2,2))
      x = hidden_layers(x,1024,strides=(1,1))
      x = GlobalAveragePooling2D()(x)
      x = Dense(64, activation='relu')(x)
      x = Dense(64, activation='relu')(x)
      output = Dense(n_classes,activation='softmax')(x)
      model = Model(input, output)
      return model
number_of_classes = 15
input_shape = (128,128,3)


def emirhan(name_model):
    print('#####~Model => {} '.format(name_model))

    model = EmirhanModel(input_shape, number_of_classes)
    print(model.summary())
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    my_callbacks = [EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=5,
                                  mode='auto')]

    history = model.fit(train_generator,
                        validation_data=test_generator,
                        epochs=64,
                        callbacks=my_callbacks,
                        verbose=0,
                        batch_size=128, )
    # Plotting Accuracy, val_accuracy, loss, val_loss
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()

    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['Train', 'Validation'])
    plt.show()

    # Predict Data Test
    pred = model.predict(test_generator)
    pred = np.argmax(pred, axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    pred = [labels[k] for k in pred]

    print('\033[01m              Classification_report \033[0m')

    print('\033[01m              Results \033[0m')
    # Results
    results = model.evaluate(test_generator, verbose=0)
    print("    Test Loss:\033[31m \033[01m {:.5f} \033[30m \033[0m".format(results[0]))
    print("Test Accuracy:\033[32m \033[01m {:.2f}% \033[30m \033[0m".format(results[1] * 100))

    return results



from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
result_VGG19 = func(preprocess_input,VGG19)


from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
result_VGG16 = func(preprocess_input,VGG16)


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
result_ResNet50 = func(preprocess_input,ResNet50)


from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
result_ResNet101 = func(preprocess_input,ResNet101)


from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
result_MobileNet = func(preprocess_input,MobileNet)


from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
result_DenseNet201 = func(preprocess_input,DenseNet201)


from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
result_Eff = func(preprocess_input,EfficientNetB7)


from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
result_Xception = func(preprocess_input,Xception)


from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
result_InResNetV2 = func(preprocess_input,InceptionResNetV2)



model_name = "Emirhan_Human_Action_Detection_with_Artificial_Intelligence"
result_emirhan = emirhan_func(model_name)

accuracy_result_table = pd.DataFrame({'Model':['Emirhan_Model','VGG16','VGG19','ResNet50','ResNet101','MobileNet','InceptionResNetV2',
                               'DenseNet201','Xception','EfficientNetB7'],
                      'Accuracy':[result_emirhan[1],result_VGG16[1], result_VGG19[1], result_ResNet50[1], result_ResNet101[1],
                                  result_MobileNet[1],result_InResNetV2[1],result_DenseNet201[1],result_Xception[1],
                                 result_Eff[1]]})

plt.figure(figsize=(12, 7))
plots = sns.barplot(x='Model', y='Accuracy', data=accuracy_result_table)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 9),
                   textcoords='offset points')

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=20);


loss_result_table = pd.DataFrame({'Model':['Emirhan_Model','VGG16','VGG19','ResNet50','ResNet101','MobileNet','InceptionResNetV2',
                               'DenseNet201','Xception','EfficientNetB7'],
                      'Loss':[result_emirhan[0],result_VGG16[0], result_VGG19[0], result_ResNet50[0], result_ResNet101[0],
                                  result_MobileNet[0],result_InResNetV2[0],result_DenseNet201[0],result_Xception[0],
                                 result_Eff[0]]})


plt.figure(figsize=(12, 7))
plots = sns.barplot(x='Model', y='Loss', data=loss_result_table)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 9),
                   textcoords='offset points')

plt.xlabel("Models")
plt.ylabel("Loss")
plt.xticks(rotation=20);