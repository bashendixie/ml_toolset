import torch
import os
# 定义根目录，后跟测试数据集路径
BASE_PATH = "dataset"
TEST_PATH = os.path.join(BASE_PATH, "test_set")
# 指定图像大小和批量大小
IMAGE_SIZE = 384
PRED_BATCH_SIZE = 4
# 确定设备类型
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# 定义保存输出的路径
OUTPUT_PATH = "output"
MIDAS_OUTPUT = os.path.join(OUTPUT_PATH, "midas_output")