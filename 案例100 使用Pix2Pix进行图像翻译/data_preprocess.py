# import the necessary packages
import tensorflow as tf

# define the module level autotune
AUTO = tf.data.AUTOTUNE

def load_image(imageFile):
	# read and decode an images file from the path
	image = tf.io.read_file(imageFile)
	image = tf.io.decode_jpeg(image, channels=3)

	# calculate the midpoint of the width and split the
	# combined images into input mask and real images
	width = tf.shape(image)[1]
	splitPoint = width // 2
	inputMask = image[:, splitPoint:, :]
	realImage = image[:, :splitPoint, :]

	# convert both images to float32 tensors and
	# convert pixels to the range of -1 and 1
	inputMask = tf.cast(inputMask, tf.float32)/127.5 - 1
	realImage = tf.cast(realImage, tf.float32)/127.5 - 1

	# return the input mask and real masks images
	return (inputMask, realImage)

def random_jitter(inputMask, realImage, height, width):
	# upscale the images for cropping purposes
	inputMask = tf.image.resize(inputMask, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	realImage = tf.image.resize(realImage, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	# return the input mask and real masks images
	return (inputMask, realImage)


class ReadTrainExample(object):
	def __init__(self, imageHeight, imageWidth):
		self.imageHeight = imageHeight
		self.imageWidth = imageWidth

	def __call__(self, imageFile):
		# read the file path and unpack the images pair
		inputMask, realImage = load_image(imageFile)

		# perform data augmentation
		(inputMask, realImage) = random_jitter(inputMask, realImage, self.imageHeight + 30, self.imageWidth + 30)

		# reshape the input mask and real masks images
		inputMask = tf.image.resize(inputMask, [self.imageHeight, self.imageWidth])
		realImage = tf.image.resize(realImage, [self.imageHeight, self.imageWidth])

		# return the input mask and real masks images
		return (inputMask, realImage)

class ReadTestExample(object):
	def __init__(self, imageHeight, imageWidth):
		self.imageHeight = imageHeight
		self.imageWidth = imageWidth

	def __call__(self, imageFile):
		# read the file path and unpack the images pair
		(inputMask, realImage) = load_image(imageFile)

		# reshape the input mask and real masks images
		inputMask = tf.image.resize(inputMask, [self.imageHeight, self.imageWidth])
		realImage = tf.image.resize(realImage, [self.imageHeight, self.imageWidth])

		# return the input mask and real masks images
		return (inputMask, realImage)

def load_dataset(path, batchSize, height, width, train=False):
	# check if this is the training dataset
	if train:
		# read the training examples
		dataset = tf.data.Dataset.list_files(str(path/"train/*.jpg"))
		dataset = dataset.map(ReadTrainExample(height, width), num_parallel_calls=AUTO)
	# otherwise, we are working with the test dataset
	else:
		# read the test examples
		dataset = tf.data.Dataset.list_files(str(path/"val/*.jpg"))
		dataset = dataset.map(ReadTestExample(height, width), num_parallel_calls=AUTO)

	# shuffle, batch, repeat and prefetch the dataset
	dataset = (dataset
		.shuffle(batchSize * 2)
		.batch(batchSize)
		.repeat()
		.prefetch(AUTO)
	)

	# return the dataset
	return dataset