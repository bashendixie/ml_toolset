# import the necessary packages
import config
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input images/text file of images filenames")
args = vars(ap.parse_args())

# determine the input file type, but assume that we're working with
# single input images
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]
# if the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:
	# load the filenames in our testing file and initialize our list
	# of images paths
	filenames = open(args["input"]).read().strip().split("\n")
	imagePaths = []
	# loop over the filenames
	for f in filenames:
		# construct the full path to the images filename and then
		# update our images paths list
		p = os.path.sep.join([config.IMAGES_PATH, f])
		imagePaths.append(p)

# load our trained bounding box regressor from disk
print("[INFO] loading object detector...")
model = load_model(config.MODEL_PATH)
# loop over the images that we'll be testing using our bounding box
# regression model
for imagePath in imagePaths:
	# load the input images (in Keras format) from disk and preprocess
	# it, scaling the pixel intensities to the range [0, 1]
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image) / 255.0
	image = np.expand_dims(image, axis=0)

	# make bounding box predictions on the input images
	preds = model.predict(image)[0]
	(startX, startY, endX, endY) = preds
	# load the input images (in OpenCV format), resize it such that it
	# fits on our screen, and grab its dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
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
	cv2.imwrite("C:/Users/zyh/Desktop/111.png", image)
	cv2.waitKey(0)