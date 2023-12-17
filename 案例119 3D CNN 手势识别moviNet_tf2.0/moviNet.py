import os
import imageio
from IPython import display

import tqdm
import random
import pathlib
import itertools
import collections
import PIL

import cv2
import einops
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint

"""
该数据集为个人录制的七种动态手势，其中每种手势各五十个动态视频。七种手势分别对应为：
0 ：点击  1： 放大  2：向下滑动  3：向上滑动  4：缩小  5：旋转  6：抓取
"""

KMP_DUPLICATE_LIB_OK = True
local_video_dir = pathlib.Path(r'D:\AI\moviNet_tf2.0\dataset')  # 本地视频数据集的文件夹路径


def format_frames(frame, output_size):
    if frame is None:
        print("Received NoneType frame")
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=5):
    result = []
    video_path = str(video_path) + ".avi"
    src = cv2.VideoCapture(video_path)
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(f"Video path: {video_path}")
    # print(f"Video frames: {video_length}")
    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)

    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result


class FrameGenerator:
    def __init__(self, path, n_frames, training=False):
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_ids_for_name = self.read_txt_file()

    def read_txt_file(self):
        txt_filename = 'mydata_train_video.txt' if self.training else 'mydata_val_video.txt'
        txt_path = self.path / txt_filename

        class_ids_for_name = {}
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                video_filename, label = parts[0], int(parts[1])
                # 从视频文件名中提取类别键
                video_key, _ = os.path.splitext(video_filename)
                class_ids_for_name[video_key] = label

        return class_ids_for_name

    def get_files_and_class_names(self):
        train_or_val = 'train' if self.training else 'val'
        txt_filename = f'mydata_{train_or_val}_video.txt'

        txt_path = self.path / txt_filename
        label_map = self.read_txt_file()

        # 构建训练或验证文件夹的完整路径
        subset_dir = self.path / train_or_val

        video_paths = [subset_dir / filename for filename in label_map.keys()]
        classes = list(label_map.values())
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, label in pairs:
            video_frames = frames_from_video_file(path, self.n_frames)
            video_filename = path.stem  # 获取文件名
            label = self.class_ids_for_name[video_filename]
            yield video_frames, label


# fg = FrameGenerator(local_video_dir, n_frames=10, training=True)
# frames, label = next(fg())
# print(f"Shape: {frames.shape}")
# print(f"Label: {label}")


# def to_gif(images, gif_path):
#     converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
#     imageio.mimsave(gif_path, converted_images, fps=10)
#
#
# video_path = local_video_dir / "train" / "dianji1"
# gif_path = str(local_video_dir / 'animation.gif')
# sample_video = frames_from_video_file(video_path, n_frames = 10)
# to_gif(sample_video, gif_path)

# 创建训练集
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
train_ds = tf.data.Dataset.from_generator(FrameGenerator(local_video_dir, 10, training=True),
                                          output_signature = output_signature)

# for frames, labels in train_ds.take(10):
#     print(labels)

# 创建验证集
val_ds = tf.data.Dataset.from_generator(FrameGenerator(local_video_dir, 10, training=False),
                                        output_signature = output_signature)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)

train_ds = train_ds.batch(8)
val_ds = val_ds.batch(8)

# train_frames, train_labels = next(iter(train_ds))
# print(f'Shape of training set of frames: {train_frames.shape}')
# print(f'Shape of training labels: {train_labels.shape}')
#
# val_frames, val_labels = next(iter(val_ds))
# print(f'Shape of validation set of frames: {val_frames.shape}')
# print(f'Shape of validation labels: {val_labels.shape}')
'''
Shape of training set of frames: (2, 10, 224, 224, 3)
Shape of training labels: (2,)
Shape of validation set of frames: (2, 10, 224, 224, 3)
Shape of validation labels: (2,)
'''

# Define the dimensions of one frame in the set of frames created
HEIGHT = 224
WIDTH = 224


class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        """
          A sequence of convolutional layers that first apply the convolution operation over the
          spatial dimensions, and then the temporal dimension.
        """
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.seq = keras.Sequential([
            # Spatial decomposition
            layers.Conv3D(filters=filters,
                          kernel_size=(1, kernel_size[1], kernel_size[2]),
                          padding=padding),
            # Temporal decomposition
            layers.Conv3D(filters=filters,
                          kernel_size=(kernel_size[0], 1, 1),
                          padding=padding)
            ])

    def call(self, x):
        return self.seq(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding
        })
        return config


class ResidualMain(keras.layers.Layer):
    """
        Residual block of the model with convolution, layer normalization, and the
        activation function, ReLU.
    """
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.seq = keras.Sequential([
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config


class Project(keras.layers.Layer):
    """
      Project certain dimensions of the tensor as the data is passed through different
      sized filters and downsampled.
    """
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.seq = keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units
        })
        return config


def add_residual_block(input, filters, kernel_size):
    """
      Add residual blocks to the model. If the last dimensions of the input data
      and filter size does not match, project it such that last dimension matches.
    """
    out = ResidualMain(filters, kernel_size)(input)

    res = input
    # Using the Keras functional APIs, project the last dimension of the tensor to
    # match the new filter size
    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)

    return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        """
        Use the einops library to resize the tensor.

        Args:
            video: Tensor representation of the video, in the form of a set of frames.

        Return:
            A downsampled size of the video according to the new height and width it should be resized to.
        """
        # b stands for batch size, t stands for time, h stands for height,
        # w stands for width, and c stands for the number of channels.
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(
            images, '(b t) h w c -> b t h w c',
            t=old_shape['t'])
        return videos

    def get_config(self):
        config = super().get_config()
        config.update({
            'height': self.height,
            'width': self.width
        })
        return config


input_shape = (None, 10, HEIGHT, WIDTH, 3)
input = layers.Input(shape=(input_shape[1:]))
x = input

x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

# Block 1
x = add_residual_block(x, 16, (3, 3, 3))
x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

# Block 2
x = add_residual_block(x, 32, (3, 3, 3))
x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

# Block 3
x = add_residual_block(x, 64, (3, 3, 3))
x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

# Block 4
x = add_residual_block(x, 128, (3, 3, 3))

x = layers.GlobalAveragePooling3D()(x)
x = layers.Flatten()(x)
x = layers.Dense(7)(x)

model = keras.Model(input, x)

# save model with best acc
# 定义保存模型的路径和文件名
model_save_path = r"D:\AI\moviNet_tf2.0\models\best_model.h5"

# 创建 ModelCheckpoint 回调函数，设置监控的指标为验证集准确率，当它有提升时保存模型
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')

frames, label = next(iter(train_ds))
model.build(frames)

# Visualize the model
keras.utils.plot_model(model, expand_nested=True, dpi=60, show_shapes=True)

model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = ['accuracy'])

history = model.fit(x = train_ds,
                    epochs = 50,
                    validation_data = val_ds,
                    callbacks=[checkpoint])


def plot_history(history):
    """
      Plotting training and validation learning curves.

      Args:
        history: model history with all the metric measures
    """
    fig, (ax1, ax2) = plt.subplots(2)

    fig.set_size_inches(18.5, 10.5)

    # Plot loss
    ax1.set_title('Loss')
    ax1.plot(history.history['loss'], label = 'train')
    ax1.plot(history.history['val_loss'], label = 'test')
    ax1.set_ylabel('Loss')

    # Determine upper bound of y-axis
    max_loss = max(history.history['loss'] + history.history['val_loss'])

    ax1.set_ylim([0, np.ceil(max_loss)])
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])

    # Plot accuracy
    ax2.set_title('Accuracy')
    ax2.plot(history.history['accuracy'],  label = 'train')
    ax2.plot(history.history['val_accuracy'], label = 'test')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])

    plt.show()


plot_history(history)

# model.evaluate(test_ds, return_dict=True)
#
#
# def get_actual_predicted_labels(dataset):
#     """
#     Create a list of actual ground truth values and the predictions from the model.
#
#     Args:
#       dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.
#
#     Return:
#       Ground truth and predicted values for a particular dataset.
#     """
#     actual = [labels for _, labels in dataset.unbatch()]
#     predicted = model.predict(dataset)
#
#     actual = tf.stack(actual, axis=0)
#     predicted = tf.concat(predicted, axis=0)
#     predicted = tf.argmax(predicted, axis=1)
#
#     return actual, predicted
#
#
# def plot_confusion_matrix(actual, predicted, labels, ds_type):
#     cm = tf.math.confusion_matrix(actual, predicted)
#     ax = sns.heatmap(cm, annot=True, fmt='g')
#     sns.set(rc={'figure.figsize':(12, 12)})
#     sns.set(font_scale=1.4)
#     ax.set_title('Confusion matrix of action recognition for ' + ds_type)
#     ax.set_xlabel('Predicted Action')
#     ax.set_ylabel('Actual Action')
#     plt.xticks(rotation=90)
#     plt.yticks(rotation=0)
#     ax.xaxis.set_ticklabels(labels)
#     ax.yaxis.set_ticklabels(labels)
#
#
# fg = FrameGenerator(local_video_dir, n_frames, training=True)
# labels = list(fg.class_ids_for_name.keys())
#
# actual, predicted = get_actual_predicted_labels(train_ds)
# plot_confusion_matrix(actual, predicted, labels, 'training')
#
# actual, predicted = get_actual_predicted_labels(test_ds)
# plot_confusion_matrix(actual, predicted, labels, 'test')
#
#
# def calculate_classification_metrics(y_actual, y_pred, labels):
#     """
#       Calculate the precision and recall of a classification model using the ground truth and
#       predicted values.
#
#       Args:
#         y_actual: Ground truth labels.
#         y_pred: Predicted labels.
#         labels: List of classification labels.
#
#       Return:
#         Precision and recall measures.
#     """
#     cm = tf.math.confusion_matrix(y_actual, y_pred)
#     tp = np.diag(cm) # Diagonal represents true positives
#     precision = dict()
#     recall = dict()
#     for i in range(len(labels)):
#         col = cm[:, i]
#         fp = np.sum(col) - tp[i] # Sum of column minus true positive is false negative
#
#         row = cm[i, :]
#         fn = np.sum(row) - tp[i] # Sum of row minus true positive, is false negative
#
#         precision[labels[i]] = tp[i] / (tp[i] + fp) # Precision
#
#         recall[labels[i]] = tp[i] / (tp[i] + fn) # Recall
#
#     return precision, recall
#
#
# precision, recall = calculate_classification_metrics(actual, predicted, labels) # Test dataset
#
# print(precision)
# print(recall)
