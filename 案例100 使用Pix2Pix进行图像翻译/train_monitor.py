# import the necessary packages
from keras.preprocessing.image import array_to_img
from keras.callbacks import Callback
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import tensorflow as tf

def get_train_monitor(testDs, imagePath, batchSize, epochInterval):
	# grab the input mask and the real images from the testing dataset
	(tInputMask, tRealImage) = next(iter(testDs))

	class TrainMonitor(Callback):
		def __init__(self, epochInterval=None):
			self.epochInterval = epochInterval

		def on_epoch_end(self, epoch, logs=None):
			if self.epochInterval and epoch % self.epochInterval == 0:
				# get the pix2pix prediction
				tPix2pixGenPred = self.model.generator.predict(tInputMask)

				(fig, axes) = subplots(nrows=batchSize, ncols=3, figsize=(50, 50))
				# plot the predicted images
				for (ax, inp, pred, tgt) in zip(axes, tInputMask, tPix2pixGenPred, tRealImage):
					# plot the input mask images
					ax[0].imshow(array_to_img(inp))
					ax[0].set_title("Input Image")

					# plot the predicted Pix2Pix images
					ax[1].imshow(array_to_img(pred))
					ax[1].set_title("Pix2Pix Prediction")

					# plot the ground truth
					ax[2].imshow(array_to_img(tgt))
					ax[2].set_title("Target Label")

				plt.savefig(f"{imagePath}/{epoch:03d}.png")
				plt.close()

	# instantiate a train monitor callback
	trainMonitor = TrainMonitor(epochInterval=epochInterval)

	# return the train monitor
	return trainMonitor