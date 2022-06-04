# 猫脸检测

import os
import sys
import glob
import dlib
import cv2

# options用于设置训练的参数和模式
options = dlib.simple_object_detector_training_options()
# Since faces are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.
options.add_left_right_image_flips = True
# 支持向量机的C参数，通常默认取为5.自己适当更改参数以达到最好的效果
options.C = 30
# 线程数，你电脑有4核的话就填4
options.num_threads = 8
options.be_verbose = True

# 获取路径
# train_folder = 'E:/dlib-master/tools/imglab/build/Release/'
# test_folder = current_path + '/cats_test/'
train_xml_path = 'E:/dlib-master/tools/imglab/build/Release/mydataset.xml'
test_xml_path = 'E:/dlib-master/tools/imglab/build/Release/testdataset.xml'

# print("training file path:" + train_xml_path)
# print(train_xml_path)
# print("testing file path:" + test_xml_path)
# print(test_xml_path)

# 开始训练
print("start training:")
dlib.train_simple_object_detector(train_xml_path, 'detector.svm', options)

#print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(train_xml_path, "detector.svm")))

print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(test_xml_path, "detector.svm")))