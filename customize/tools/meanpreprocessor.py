# import the necessary packages
import cv2

class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        # store the Red, Green, and Blue channel averages across a
        # training set
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        # split the images into its respective Red, Green, and Blue
        # channels
        (B, G, R) = cv2.split(image.astype("float32"))

        # subtract the means for each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        # merge the channels back together and return the images
        return cv2.merge([B, G, R])