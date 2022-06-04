import numpy as np
import tensorflow as tf
from fcn8s import *
#from fcn8s import FCN8s
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

# TestSet  = DataGenerator("./data/test_image.txt", "./data/test_labels", 1)
# model = load_model("FCN8s.h5")
# results = model.predict_generator(TestSet, 1, verbose=1)
# for i,item in enumerate(results):
#         img = labelVisualize(21, colormap, item) #if flag_multi_class else item[:, :, 0]
#         io.imsave(os.path.join("data/prediction", "%d_predict.png"%i), img)




#model = FCN8s(n_class=21)
TestSet  = DataGenerator("./data/test_image.txt", "./data/test_labels", 1)

## load weights and test your model after training
## if you want to test model, first you need to initialize your model
## with "model(data)", and then load model weights
data = np.ones(shape=[1,224,224,3], dtype=np.float)
model(data)
model.load_weights("FCN8s.h5")


results = model.predict_generator(TestSet, 1, verbose=1)
pred_label = tf.argmax(results, axis=-1)
img = labelVisualize(21, colormap, pred_label[0].numpy())
io.imsave(os.path.join("data/prediction", "predict.png"), img)

# for i,item in enumerate(results):
#         img = labelVisualize(21, colormap, item) #if flag_multi_class else item[:, :, 0]
#         io.imsave(os.path.join("data/prediction", "%d_predict.png"%i), item[0].numpy())


# for idx, (x, y) in enumerate(TestSet):
#     result = model(x)
#     pred_label = tf.argmax(result, axis=-1)
#     result = visual_result(x[0], pred_label[0].numpy())
#     save_file = "./data/prediction/%d.jpg" %idx
#     print("=> saving prediction result into ", save_file)
#     result.save(save_file)
#     if idx == 209:
#         result.show()
#         break