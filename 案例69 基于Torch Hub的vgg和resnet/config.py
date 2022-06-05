import torch
import os

# 定义数据目录，训练和测试路径
BASE_PATH = "dataset"
TRAIN_PATH = os.path.join(BASE_PATH, "training_set")
TEST_PATH = os.path.join(BASE_PATH, "test_set")

# 指定 ImageNet 均值和标准差
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 指定训练超参数
IMAGE_SIZE = 256
BATCH_SIZE = 8
PRED_BATCH_SIZE = 8
EPOCHS = 2
LR = 0.0001

# 确定设备类型
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# 定义存储训练图和训练模型的路径
PLOT_PATH = os.path.join("output", "model_training.png")
MODEL_PATH = os.path.join("output", "model.pth")



