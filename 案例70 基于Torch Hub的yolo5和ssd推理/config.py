import torch
import os
# 定义根目录，后跟测试数据集路径
BASE_PATH = "dataset"
TEST_PATH = os.path.join(BASE_PATH, "test_set")
# 指定图像大小和批量大小
IMAGE_SIZE = 300
PRED_BATCH_SIZE = 4
# 指定 ssd 检测的阈值置信度值
THRESHOLD = 0.50
# 确定设备类型
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# 定义保存输出的路径
OUTPUT_PATH = "output"
SSD_OUTPUT = os.path.join(OUTPUT_PATH, "ssd_output")
YOLO_OUTPUT = os.path.join(OUTPUT_PATH, "yolo_output")

