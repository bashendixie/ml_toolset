# import OpenCV before mxnet to avoid a segmentation fault
import cv2

# import the necessary packages
import age_gender_config as config
import age_gender_deploy as deploy
from customize.tools.imagetoarraypreprocessor import ImageToArrayPreprocessor
from customize.tools.simplepreprocessor import SimplePreprocessor
from customize.tools.meanpreprocessor import MeanPreprocessor
from agegenderhelper import AgeGenderHelper
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import json
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sample-size", type=int, default=10, help="epoch # to load")
args = vars(ap.parse_args())

# load the label encoders and mean files
print("[INFO] loading label encoders and mean files...")
ageLE = pickle.loads(open(deploy.AGE_LABEL_ENCODER, "rb").read())
genderLE = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, "rb").read())
ageMeans = json.loads(open(deploy.AGE_MEANS).read())
genderMeans = json.loads(open(deploy.GENDER_MEANS).read())

# load the models from disk
print("[INFO] loading models...")
agePath = os.path.sep.join([deploy.AGE_NETWORK_PATH,
deploy.AGE_PREFIX])
genderPath = os.path.sep.join([deploy.GENDER_NETWORK_PATH,
deploy.GENDER_PREFIX])
ageModel = mx.model.FeedForward.load(agePath, deploy.AGE_EPOCH)
genderModel = mx.model.FeedForward.load(genderPath,
deploy.GENDER_EPOCH)

# now that the networks are loaded, we need to compile them
print("[INFO] compiling models...")
ageModel = mx.model.FeedForward(ctx=[mx.gpu(0)],
symbol=ageModel.symbol, arg_params=ageModel.arg_params,
aux_params=ageModel.aux_params)
genderModel = mx.model.FeedForward(ctx=[mx.gpu(0)],
symbol=genderModel.symbol, arg_params=genderModel.arg_params,
aux_params=genderModel.aux_params)

# initialize the image pre-processors
sp = SimplePreprocessor(width=227, height=227, inter=cv2.INTER_CUBIC)
ageMP = MeanPreprocessor(ageMeans["R"], ageMeans["G"],
ageMeans["B"])
genderMP = MeanPreprocessor(genderMeans["R"], genderMeans["G"],
genderMeans["B"])
iap = ImageToArrayPreprocessor()

# load a sample of testing images
rows = open(config.TEST_MX_LIST).read().strip().split("\n")
rows = np.random.choice(rows, size=args["sample_size"])

# loop over the rows
for row in rows:
    # unpack the row
    (_, gtLabel, imagePath) = row.strip().split("\t")
    image = cv2.imread(imagePath)

    # pre-process the image, one for the age model and another for
    # the gender model
    ageImage = iap.preprocess(ageMP.preprocess(
    sp.preprocess(image)))
    genderImage = iap.preprocess(genderMP.preprocess(
    sp.preprocess(image)))
    ageImage = np.expand_dims(ageImage, axis=0)
    genderImage = np.expand_dims(genderImage, axis=0)

    # pass the ROIs through their respective models
    agePreds = ageModel.predict(ageImage)[0]
    genderPreds = genderModel.predict(genderImage)[0]
    # sort the predictions according to their probability
    ageIdxs = np.argsort(agePreds)[::-1]
    genderIdxs = np.argsort(genderPreds)[::-1]

    # visualize the age and gender predictions
    ageCanvas = AgeGenderHelper.visualizeAge(agePreds, ageLE)
    genderCanvas = AgeGenderHelper.visualizeGender(genderPreds,genderLE)
    image = imutils.resize(image, width=400)

    # draw the actual prediction on the image
    gtLabel = ageLE.inverse_transform(int(gtLabel))
    text = "Actual: {}-{}".format(*gtLabel.split("_"))
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
    # show the output image
    cv2.imshow("Image", image)
    cv2.imshow("Age Probabilities", ageCanvas)
    cv2.imshow("Gender Probabilities", genderCanvas)
    cv2.waitKey(0)