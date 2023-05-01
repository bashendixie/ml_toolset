# import the necessary packages
from sklearn.feature_extraction.image import extract_patches_2d

class PatchPreprocessor:
    def __init__(self, width, height):
        # store the target width and height of the images
        self.width = width
        self.height = height

    def preprocess(self, image):
        # extract a random crop from the images with the target width
        # and height
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]