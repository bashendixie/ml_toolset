# import the necessary packages
import os

# name of the dataset we will be using
DATASET = "cityscapes"

# build the dataset URL
DATASET_URL = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{DATASET}.tar.gz"

# define the batch size
TRAIN_BATCH_SIZE = 32
INFER_BATCH_SIZE = 8

# dataset specs
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNELS = 3

# training specs
LEARNING_RATE = 2e-4
EPOCHS = 150
STEPS_PER_EPOCH = 100

# path to our base output directory
BASE_OUTPUT_PATH = "outputs"
BASE_IMAGE_PATH = ""

# GPU training pix2pix model paths
GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH, "models", "generator")

# define the path to the inferred images and to the grid images
BASE_IMAGES_PATH = os.path.join(BASE_OUTPUT_PATH, "images")
GRID_IMAGE_PATH = os.path.join(BASE_IMAGE_PATH, "grid.png")