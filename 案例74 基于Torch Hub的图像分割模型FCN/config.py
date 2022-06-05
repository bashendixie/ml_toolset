import os
import torch
# 定义 cuda 或 cpu 使用
DEVICE = "cpu" #torch.device("cuda") if torch.cuda.is_available() else "cpu"
# 定义根目录，后跟测试数据集路径
BASE_PATH = "dataset"
TEST_PATH = os.path.join(BASE_PATH, "test_set")
# 定义预训练模型名称和训练的类数
MODEL = ["fcn_resnet50", "fcn_resnet101"]
NUM_CLASSES = 21
# 指定图像大小和批量大小
IMAGE_SIZE = 224
BATCH_SIZE = 4
# 定义基本输出目录的路径
BASE_OUTPUT = "output"
# 定义输入图像的路径和输出分割掩码可视化
SAVE_IMAGE_PATH = os.path.join(BASE_OUTPUT, "image_samples.png")
SEGMENTATION_OUTPUT_PATH = os.path.sep.join([BASE_OUTPUT,
	"segmentation_output.png"])