import numpy as np
import tensorflow as tf
from fcn8s import *
from utils import visual_result, DataGenerator, colormap
import skimage.io as io

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

def labelVisualize(num_class, color_dict, img):
    H, W = img.shape
    masks_color = np.zeros(shape=[H, W, 3])
    for i in range(H):
        for j in range(W):
            cls_idx = img[i, j]
            masks_color[i, j] = color_dict[cls_idx]
    return masks_color

TestSet  = DataGenerator("./data/test_image.txt", "./data/test_labels", 1)
data = np.ones(shape=[1,224,224,3], dtype=np.float)
model(data)
model.load_weights("FCN8s.h5")
results = model.predict_generator(TestSet, 1, verbose=1)
pred_label = tf.argmax(results, axis=-1)
img = labelVisualize(21, colormap, pred_label[0].numpy())
io.imsave(os.path.join("data/prediction", "predict.png"), img)