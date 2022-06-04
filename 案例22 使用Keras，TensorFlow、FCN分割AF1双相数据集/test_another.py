
from model_8s import *
from data import *
import matplotlib
import os
from keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def test():
    #加载模型h5文件
    model = load_model("fcn32_another.hdf5")
    testGene = testGenerator_another("data/test_another", 1, as_gray=False)
    results = model.predict_generator(testGene, 1, verbose=1)
    saveResult("data/result_another", results)

test()