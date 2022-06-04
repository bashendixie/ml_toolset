from model import *
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
    model = load_model("unet_meiyan.hdf5")
    testGene = testGenerator("data/membrane/test_m")
    results = model.predict_generator(testGene, 18, verbose=1)
    saveResult("data/membrane/result_m", results)


def h5_to_pb():
    model = tf.keras.models.load_model('unet_meiyan.hdf5',
                                       custom_objects={'KerasLayer': hub.KerasLayer, 'Dense': tf.keras.layers.Dense},
                                       compile=False)
    model.summary()
    full_model = tf.function(lambda Input: model(Input))

    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, tf.float32))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="D:\\", name="unet_meiyan.pb", as_text=False)


test()