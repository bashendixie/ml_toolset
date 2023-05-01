# import the necessary packages
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import VGG16, EfficientNetB3
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
import tensorflow as tf
import numpy as np
import argparse
import os
import imutils
import cv2


class Generator(tf.keras.utils.Sequence):
    def __init__(self, DATASET_PATH, BATCH_SIZE=32, shuffle_images=True, image_min_side=24):
        self.batch_size = BATCH_SIZE
        self.shuffle_images = shuffle_images
        self.image_min_side = image_min_side
        self.load_image_paths_labels(DATASET_PATH)
        self.create_image_groups()

    def load_image_paths_labels(self, DATASET_PATH):
        classes = os.listdir(DATASET_PATH)
        lb = preprocessing.LabelBinarizer()
        lb.fit(classes)
        self.image_paths = []
        self.image_labels = []
        for class_name in classes:
            class_path = os.path.join(DATASET_PATH, class_name)
            for image_file_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, image_file_name))
                self.image_labels.append(class_name)

        self.image_labels = np.array(lb.transform(self.image_labels), dtype='float32')

        assert len(self.image_paths) == len(self.image_labels)

    def create_image_groups(self):
        if self.shuffle_images:
            # Randomly shuffle dataset
            seed = 4321
            np.random.seed(seed)
            np.random.shuffle(self.image_paths)
            np.random.seed(seed)
            np.random.shuffle(self.image_labels)

        # Divide image_paths and image_labels into groups of BATCH_SIZE
        self.image_groups = [[self.image_paths[x % len(self.image_paths)] for x in range(i, i + self.batch_size)]
                             for i in range(0, len(self.image_paths), self.batch_size)]
        self.label_groups = [[self.image_labels[x % len(self.image_labels)] for x in range(i, i + self.batch_size)]
                             for i in range(0, len(self.image_labels), self.batch_size)]

    def resize_image(self, img, min_side_len):

        h, w, c = img.shape

        # limit the min side maintaining the aspect ratio
        if min(h, w) < min_side_len:
            im_scale = float(min_side_len) / h if h < w else float(min_side_len) / w
        else:
            im_scale = 1.

        new_h = 224 #int(h * im_scale)
        new_w = 224 #int(w * im_scale)

        re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return re_im, new_h / h, new_w / w

    def load_images(self, image_group):

        images = []
        for image_path in image_group:
            img = cv2.imread(image_path)
            img_shape = len(img.shape)
            if img_shape == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img_shape == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img_shape == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, rh, rw = self.resize_image(img, self.image_min_side)
            images.append(img)

        return images

    def construct_image_batch(self, image_group):
        # get the max images shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an images batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype='float32')

        # copy all images to the upper left part of the images batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.image_groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        image_group = self.image_groups[index]
        label_group = self.label_groups[index]
        images = self.load_images(image_group)
        image_batch = self.construct_image_batch(images)

        return np.array(image_batch), np.array(label_group)


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the images preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the images and extract the class masks assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{images}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the images
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our processed images as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))

class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target images width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # grab the dimensions of the images and then initialize
        # the deltas to use when cropping
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0
        # if the width is smaller than the height, then resize
        # along the width (i.e., the smaller dimension) and then
        # update the deltas to crop the height to the desired
        # dimension
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        # otherwise, the height is smaller than the width so
        # resize along the height and then update the deltas
        # to crop along the width
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        # now that our images have been resized, we need to
        # re-grab the width and height, followed by performing
        # the crop
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        # finally, resize the images to the provided spatial
        # dimensions to ensure our output images is always a fixed
        # size
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

class ImageToArrayPreprocessor:
	def __init__(self, dataFormat=None):
		# store the images data format
		self.dataFormat = dataFormat
	def preprocess(self, image):
		# apply the Keras utility function that correctly rearranges
		# the dimensions of the images
		return img_to_array(image, data_format=self.dataFormat)

class FCHeadNet:
    @staticmethod
    def build(baseModel, classes, D):
        # initialize the head model that will be placed on top of
        # the base, then add a FC layer
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)

        # add a softmax layer
        headModel = Dense(classes, activation="softmax")(headModel)

        # return the model
        return headModel


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
#ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

BATCH_SIZE = 16
train_dir = 'data_set/train'
val_dir = 'data_set/val'
train_generator = Generator(train_dir, BATCH_SIZE, shuffle_images=True, image_min_side=24)
val_generator = Generator(val_dir, BATCH_SIZE, shuffle_images=True, image_min_side=24)

# 加载 VGG16 网络，确保头部 FC 层集被关闭
baseModel = EfficientNetB3(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# 初始化网络的新头，一组 FC 层，后跟一个 softmax 分类器
headModel = FCHeadNet.build(baseModel, 10, 256)

# 将头部 FC 模型放在基础模型之上——这将成为我们将训练的实际模型
model = Model(inputs=baseModel.input, outputs=headModel)

# 遍历基础模型中的所有层并冻结它们，以便它们在训练过程中*不会*更新
for layer in baseModel.layers:
    layer.trainable = False

# 编译我们的模型（这需要在我们将层设置为不可训练之后完成
print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

checkpoint_path = './snapshots'
os.makedirs(checkpoint_path, exist_ok=True)
model_path = os.path.join(checkpoint_path, 'model_epoch_vgg16.h5')

# train the head of the network for a few epochs (all other
# layers are frozen) -- this will allow the new FC layers to
# start to become initialized with actual "learned" values
# versus pure random
print("[INFO] training head...")
model.fit_generator(generator=train_generator,
                    steps_per_epoch=len(train_generator),
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    callbacks=[tf.keras.callbacks.ModelCheckpoint(model_path,
                                                                  monitor='val_loss',
                                                                  save_best_only=True, verbose=1)],
                    epochs=225,
                    verbose=1)


# # 现在头部 FC 层已经训练/初始化，让我们解冻最终的 CONV 层集并使它们可训练
# for layer in baseModel.layers[15:]:
#     layer.trainable = True
#
# # 为了使模型的更改生效，我们需要重新编译模型，这次使用具有非常小学习率的 SGD
# print("[INFO] re-compiling model...")
# opt = SGD(lr=0.001)
# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#
#
#
# # 再次训练模型，这次微调*两个*最终的 CONV 层集以及我们的 FC 层集
# print("[INFO] fine-tuning model...")
# model.fit_generator(generator=train_generator,
#                     steps_per_epoch=len(train_generator),
#                     validation_data=val_generator,
#                     validation_steps=len(val_generator),
#                     callbacks=[tf.keras.callbacks.ModelCheckpoint(model_path,
#                                                                   monitor='val_loss',
#                                                                   save_best_only=True, verbose=1)],
#                     epochs=100, verbose=1)

# 在微调模型上评估网络
# print("[INFO] evaluating after fine-tuning...")
# predictions = model.predict(testX, batch_size=32)
# print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# 将模型保存到磁盘
# print("[INFO] serializing model...")
# model.save(args["model"])

