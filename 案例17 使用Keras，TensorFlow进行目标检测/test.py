# import the necessary packages
import config
from keras.utils import img_to_array
from keras.utils import load_img
from keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os
import time

model = load_model(config.MODEL_PATH)

image = load_img("C:/Users/zyh/Desktop/23.png", target_size=(224, 224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

start_time = time.time()

# make bounding box predictions on the input images
preds = model.predict(image)

end_time = time.time()

print("运行时间：" + str(end_time - start_time))

pred = preds[0]
(startX, startY, endX, endY) = pred

# load the input images (in OpenCV format), resize it such that it
# fits on our screen, and grab its dimensions
image = cv2.imread("C:/Users/zyh/Desktop/23.png")
#images = imutils.resize(images, width=663)
(h, w) = image.shape[:2]
# scale the predicted bounding box coordinates based on the images
# dimensions
startX = int(startX * w)
startY = int(startY * h)
endX = int(endX * w)
endY = int(endY * h)
# draw the predicted bounding box on the images
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
# show the output images
cv2.imwrite("C:/Users/zyh/Desktop/1031.png", image)
cv2.waitKey(0)