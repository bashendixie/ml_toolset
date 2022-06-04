import cv2
import skimage.io as io
import os
import argparse
import numpy as np
from utils import colormap
from scipy import misc

if not os.path.exists("./data"): os.mkdir("./data")
if not os.path.exists("./data/train_labels"): os.mkdir("./data/train_labels")
if not os.path.exists("./data/test_labels"): os.mkdir("./data/test_labels")
if not os.path.exists("./data/prediction"): os.mkdir("./data/prediction")

parser = argparse.ArgumentParser()
parser.add_argument("--voc_path", type=str, default="D:\\deeplearn\\dataset\\data_road\\training")
flags = parser.parse_args()
if not os.path.exists(flags.voc_path): # "/home/yang/dataset/VOC"
    raise ValueError("Path: %s does not exist" %flags.voc_path)


image_write = open(os.path.join(os.getcwd(), "data\\train_image.txt"), "w")
train_label_folder = os.path.join(flags.voc_path, "gt_image_2")
train_image_folder = os.path.join(flags.voc_path, "image_2")
train_label_images = os.listdir(train_label_folder)

for train_label_image in train_label_images:
    label_name = train_label_image[:-4]
    image_path = os.path.join(train_image_folder, label_name.replace('_road', '') + ".png")
    if not os.path.exists(image_path): continue
    image_write.writelines(image_path+"\n")
    label_path = os.path.join(train_label_folder, train_label_image)
    label_image = np.array(cv2.imread(label_path, cv2.IMREAD_COLOR))
    label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
    write_label = open(("./data/train_labels/")+label_name+".txt", 'w')
    print("=> processing %s" %label_path)
    H, W, C = label_image.shape
    for i in range(H):
        write_line = []
        for j in range(W):
            pixel_color = label_image[i, j].tolist()
            if pixel_color in colormap:
                cls_idx = colormap.index(pixel_color)
            else:
                cls_idx = 0
            write_line.append(str(cls_idx))
        write_label.writelines(" ".join(write_line) + "\n")

