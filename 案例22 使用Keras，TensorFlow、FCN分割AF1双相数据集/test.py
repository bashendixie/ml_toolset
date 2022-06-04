from tensorflow.core.protobuf import saved_model_pb2
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
    model = load_model("fcn32_af1.hdf5")
    testGene = testGenerator("data/test1", 25)
    results = model.predict_generator(testGene, 25, verbose=1)
    saveResult("data/result1", results)


def h5_to_pb():
    model = tf.keras.models.load_model('fcn32_another.hdf5', custom_objects={'KerasLayer': hub.KerasLayer, 'Dense': tf.keras.layers.Dense},
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
                      logdir="D:\\", name="fcn32_another.pb", as_text=False)

def aaa():
    pb_file = 'fcn32_af1.pb'
    graph_def = tf.compat.v1.GraphDef()

    try:
        with tf.io.gfile.GFile(pb_file, 'rb') as f:
            graph_def.ParseFromString(f.read())
    except:
        with tf.gfile.FastGFile(pb_file, 'rb') as f:
            graph_def.ParseFromString(f.read())

    # Delete weights
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].op == 'Const':
            del graph_def.node[i]
    graph_def.library.Clear()
    tf.compat.v1.train.write_graph(graph_def, "", 'fcn32_af1.pbtxt', as_text=True)

test()
#h5_to_pb()
#aaa()