# import necessary packages
from data_utils import get_dataloader
import config as config
from torchvision.transforms import Compose, ToTensor, Resize
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import cv2
import os

# initialize test transform pipeline
testTransform = Compose([
	Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), ToTensor()])
# create the test dataset
testDataset = ImageFolder(config.TEST_PATH, testTransform)
# initialize the test data loader
testLoader = get_dataloader(testDataset, config.PRED_BATCH_SIZE)

# initialize the yolov5 using torch hub
yoloModel = torch.hub.load("ultralytics/yolov5", "yolov5s")
# initialize iterable variable
sweeper = iter(testLoader)
# initialize images
imageInput = []
# grab a batch of test data
print("[INFO] getting the test data...")
batch = next(sweeper)
(images, _) = (batch[0], batch[1])
# send the images to the device
images = images.to(config.DEVICE)

# loop over all the batch
for index in range(0, config.PRED_BATCH_SIZE):
	# grab each images
	# rearrange dimensions to channel last and
	# append them to images list
	image = images[index]
	image = image.permute((1, 2, 0))
	imageInput.append(image.cpu().detach().numpy()*255.0)
# pass the images list through the model
print("[INFO] getting detections from the test data...")
results = yoloModel(imageInput, size=300)

# get random index value
randomIndex = random.randint(0,len(imageInput)-1)
# grab index result from results variable
imageIndex= results.pandas().xyxy[randomIndex]
# convert the bounding box values to integer
startX = int(imageIndex["xmin"][0])
startY = int(imageIndex["ymin"][0])
endX = int(imageIndex["xmax"][0])
endY = int(imageIndex["ymax"][0])
# draw the predicted bounding box and class masks on the images
y = startY - 10 if startY - 10 > 10 else startY + 10
cv2.putText(imageInput[randomIndex], imageIndex["name"][0],
	(startX, y+10), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 255, 0), 2)
cv2.rectangle(imageInput[randomIndex],
	(startX, startY), (endX, endY),(0, 255, 0), 2)
# check to see if the output directory already exists, if not
# make the output directory
if not os.path.exists(config.YOLO_OUTPUT):
    os.makedirs(config.YOLO_OUTPUT)
# show the output images and save it to path
plt.imshow(imageInput[randomIndex]/255.0)
# save plots to output directory
print("[INFO] saving the inference...")
outputFileName = os.path.join(config.YOLO_OUTPUT, "output.png")
plt.savefig(outputFileName)