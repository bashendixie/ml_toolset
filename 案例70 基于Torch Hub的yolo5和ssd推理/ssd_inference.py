# import the necessary packages
from data_utils import get_dataloader
from data_utils import normalize
import config as config
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import cv2
import os
# initialize test transform pipeline
testTransform = Compose([
	Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), ToTensor()])
# create the test dataset and initialize the test data loader
testDataset = ImageFolder(config.TEST_PATH, testTransform)
testLoader = get_dataloader(testDataset, config.PRED_BATCH_SIZE)
# initialize iterable variable
sweeper = iter(testLoader)
# list to store permuted images
imageInput = []

# grab a batch of test data
print("[INFO] getting the test data...")
batch = next(sweeper)
(images, _) = (batch[0], batch[1])
# switch off autograd
with torch.no_grad():
	# send the images to the device
	images = images.to(config.DEVICE)

	# loop over all the batch
	for index in range(0, config.PRED_BATCH_SIZE):
		# grab the image, de-normalize it, scale the raw pixel
		# intensities to the range [0, 255], and change the channel
		# ordering from channels first tp channels last
		image = images[index]
		image = image.permute((1, 2, 0))
		imageInput.append(image.cpu().detach().numpy())

# call the required entry points
ssdModel = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
# flash model to the device and set it to eval mode
ssdModel.to(config.DEVICE)
ssdModel.eval()
# new list for processed input
processedInput = []
# loop over images and preprocess them
for image in imageInput:
	image = normalize (image)
	processedInput.append(image)
# convert the preprocessed images into tensors
inputTensor = utils.prepare_tensor(processedInput)

# turn off auto-grad
print("[INFO] getting detections from the test data...")
with torch.no_grad():
	# feed images to model
	detections = ssdModel(inputTensor)
# decode the results and filter them using the threshold
resultsPerInput = utils.decode_results(detections)
bestResults = [utils.pick_best(results,
	config.THRESHOLD) for results in resultsPerInput]

# get coco labels
classesToLabels = utils.get_coco_object_dictionary()
# loop over the image batch
for image_idx in range(len(bestResults)):
	(fig, ax) = plt.subplots(1)
	# denormalize the image and plot the image
	image = processedInput[image_idx] / 2 + 0.5
	ax.imshow(image)
	# grab bbox, class, and confidence values
	(bboxes, classes, confidences) = bestResults[image_idx]

	# loop over the detected bounding boxes
	for idx in range(len(bboxes)):
		# scale values up according to image size
		(left, bot, right, top) = bboxes[idx ] * 300
		# draw the bounding box on the image
		(x, y, w, h) = [val for val in [left, bot, right - left,
			top - bot]]
		rect = patches.Rectangle((x, y), w, h, linewidth=1,
			edgecolor="r", facecolor="none")
		ax.add_patch(rect)
		ax.text(x, y,
			"{} {:.0f}%".format(classesToLabels[classes[idx] - 1],
			confidences[idx] * 100),
			bbox=dict(facecolor="white", alpha=0.5))

# 检查输出目录是否已经存在，如果不存在则制作输出目录
if not os.path.exists(config.SSD_OUTPUT):
    os.makedirs(config.SSD_OUTPUT)
# 将绘图保存到输出目录
print("[INFO] saving the inference...")
outputFileName = os.path.join(config.SSD_OUTPUT, "output.png")
plt.savefig(outputFileName)