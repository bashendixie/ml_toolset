# import OpenCV before mxnet to avoid a segmentation fault
import cv2

# import the necessary packages
import age_gender_deploy as deploy
from customize.tools.imagetoarraypreprocessor import ImageToArrayPreprocessor
from customize.tools.simplepreprocessor import SimplePreprocessor
from customize.tools.meanpreprocessor import MeanPreprocessor
from customize.tools.croppreprocessor import CropPreprocessor
from agegenderhelper import AgeGenderHelper
from imutils.face_utils import FaceAligner
from imutils import face_utils
from imutils import paths
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import json
import dlib
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to input images (or directory)")
args = vars(ap.parse_args())

# load the masks encoders and mean files
print("[INFO] loading masks encoders and mean files...")
ageLE = pickle.loads(open(deploy.AGE_LABEL_ENCODER, "rb").read())
genderLE = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, "rb").read())
ageMeans = json.loads(open(deploy.AGE_MEANS).read())
genderMeans = json.loads(open(deploy.GENDER_MEANS).read())

# load the models from disk
print("[INFO] loading models...")
agePath = os.path.sep.join([deploy.AGE_NETWORK_PATH,deploy.AGE_PREFIX])
genderPath = os.path.sep.join([deploy.GENDER_NETWORK_PATH,deploy.GENDER_PREFIX])
ageModel = mx.model.FeedForward.load(agePath, deploy.AGE_EPOCH)
genderModel = mx.model.FeedForward.load(genderPath,deploy.GENDER_EPOCH)

# now that the networks are loaded, we need to compile them
print("[INFO] compiling models...")
ageModel = mx.model.FeedForward(ctx=[mx.gpu(0)],
symbol=ageModel.symbol, arg_params=ageModel.arg_params, aux_params=ageModel.aux_params)
genderModel = mx.model.FeedForward(ctx=[mx.gpu(0)],
symbol=genderModel.symbol, arg_params=genderModel.arg_params, aux_params=genderModel.aux_params)

# initialize the images pre-processors
sp = SimplePreprocessor(width=256, height=256, inter=cv2.INTER_CUBIC)
cp = CropPreprocessor(width=227, height=227, horiz=True)
ageMP = MeanPreprocessor(ageMeans["R"], ageMeans["G"], ageMeans["B"])
genderMP = MeanPreprocessor(genderMeans["R"], genderMeans["G"], genderMeans["B"])
iap = ImageToArrayPreprocessor(dataFormat="channels_first")

# initialize dlib’s face detector (HOG-based), then create the
# the facial landmark predictor and face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
fa = FaceAligner(predictor)

# initialize the list of images paths as just a single images
imagePaths = [args["images"]]

# if the input path is actually a directory, then list all images
# paths in the directory
if os.path.isdir(args["images"]):
    imagePaths = sorted(list(paths.list_files(args["images"])))

# loop over the images paths
for imagePath in imagePaths:
    # load the images from disk, resize it, and convert it to
    # grayscale
    print("[INFO] processing {}".format(imagePath))
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale images
    rects = detector(gray, 1)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # align the face
        shape = predictor(gray, rect)
        face = fa.align(image, gray, rect)
        # resize the face to a fixed size, then extract 10-crop
        # patches from it
        face = sp.preprocess(face)
        patches = cp.preprocess(face)
        # allocate memory for the age and gender patches
        agePatches = np.zeros((patches.shape[0], 3, 227, 227), dtype = "float")
        genderPatches = np.zeros((patches.shape[0], 3, 227, 227), dtype = "float")

        # loop over the patches
        for j in np.arange(0, patches.shape[0]):
            # perform mean subtraction on the patch
            agePatch = ageMP.preprocess(patches[j])
            genderPatch = genderMP.preprocess(patches[j])
            agePatch = iap.preprocess(agePatch)
            genderPatch = iap.preprocess(genderPatch)
            # update the respective patches lists
            agePatches[j] = agePatch
            genderPatches[j] = genderPatch

        # make predictions on age and gender based on the extracted
        # patches
        agePreds = ageModel.predict(agePatches)
        genderPreds = genderModel.predict(genderPatches)
        # compute the average for each class masks based on the
        # predictions for the patches
        agePreds = agePreds.mean(axis=0)
        genderPreds = genderPreds.mean(axis=0)

        # visualize the age and gender predictions
        ageCanvas = AgeGenderHelper.visualizeAge(agePreds, ageLE)
        genderCanvas = AgeGenderHelper.visualizeGender(genderPreds, genderLE)
        # draw the bounding box around the face
        clone = image.copy()
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        141  # show the output images
        cv2.imshow("Input", clone)
        cv2.imshow("Face", face)
        cv2.imshow("Age Probabilities", ageCanvas)
        cv2.imshow("Gender Probabilities", genderCanvas)
        cv2.waitKey(0)