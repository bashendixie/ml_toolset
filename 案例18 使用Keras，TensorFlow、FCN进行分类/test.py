# import the necessary packages
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from tensorflow.python.keras.models import *

image = cv2.imread('C:/Users/zyh/Desktop/aa.jpg')

# pre-process the image for classification
image = cv2.resize(image, (64, 64))
#image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model('C:/Users/zyh/Desktop/model_epoch.h5')

# classify the input image
santa = model.predict(image)
print(santa)