import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU = True if DEVICE == "cuda" else False
# 定义要生成和插值的图像数量
NUM_IMAGES = 8
NUM_INTERPOLATION = 8
# 定义基本输出目录的路径
BASE_OUTPUT = "output"
# 定义输出模型输出和潜在空间插值的路径
SAVE_IMG_PATH = os.path.join(BASE_OUTPUT, "image_samples.png")
INTERPOLATE_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "interpolate.png"])

