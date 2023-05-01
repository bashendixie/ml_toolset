# import the necessary packages
import os
# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file
BASE_PATH = "C:/Users/zyh/Desktop/labelImg"
IMAGES_PATH = os.path.sep.join([BASE_PATH, ""])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "train_peaches.csv"])
# define the path to the base output directory
BASE_OUTPUT = "C:/Users/zyh/Desktop/labelImg/output"
# define the path to the output serialized model, model training plot,
# and testing images filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32

