# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
	def __init__(self, dataFormat=None):
		# store the images data format
		self.dataFormat = dataFormat
	def preprocess(self, image):
		# apply the Keras utility function that correctly rearranges
		# the dimensions of the images
		return img_to_array(image, data_format=self.dataFormat)
        