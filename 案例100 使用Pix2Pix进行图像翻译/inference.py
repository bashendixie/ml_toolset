import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
import config
from data_preprocess import load_dataset
from keras.preprocessing.image import array_to_img
from keras.models import load_model
from keras.utils.all_utils import get_file
from matplotlib.pyplot import subplots
import pathlib
import os

# download the cityscape training dataset
print("[INFO] downloading the dataset...")
pathToZip = get_file(
	fname=f"{config.DATASET}.tar.gz",
	origin=config.DATASET_URL,
	extract=True
)
pathToZip  = pathlib.Path(pathToZip)
path = pathToZip.parent/config.DATASET

# build the test dataset
print("[INFO] building the test dataset...")
testDs = load_dataset(path=path, train=False, batchSize=config.INFER_BATCH_SIZE, height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH)

# get the first batch of testing images
(inputMask, realImage) = next(iter(testDs))

# set the path for the generator
genPath = config.GENERATOR_MODEL

# load the trained pix2pix generator
print("[INFO] loading the trained pix2pix generator...")
pix2pixGen = load_model(genPath, compile=False)

# predict using pix2pix generator
print("[INFO] making predictions with the generator...")
pix2pixGenPred = pix2pixGen.predict(inputMask)

# plot the respective predictions
print("[INFO] saving the predictions...")
(fig, axes) = subplots(nrows=config.INFER_BATCH_SIZE, ncols=3, figsize=(50, 50))

# plot the predicted images
for (ax, inp, pred, tar) in zip(axes, inputMask, pix2pixGenPred, realImage):
	# plot the input mask images
	ax[0].imshow(array_to_img(inp))
	ax[0].set_title("Input Image")

	# plot the predicted Pix2Pix images
	ax[1].imshow(array_to_img(pred))
	ax[1].set_title("pix2pix prediction")

	# plot the ground truth
	ax[2].imshow(array_to_img(tar))
	ax[2].set_title("Target masks")

# check whether output images directory exists, if it doesn't, then
# create it
if not os.path.exists(config.BASE_IMAGEs_PATH):
	os.makedirs(config.BASE_IMAGES_PATH)

# serialize the results to disk
print("[INFO] saving the pix2pix predictions to disk...")
fig.savefig(config.GRID_IMAGE_PATH)