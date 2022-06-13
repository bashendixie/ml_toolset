import os

# 初始化图像的原始目录的路径
ORIG_INPUT_DATASET = "datasets/orig"

# 在计算训练和测试拆分后，初始化新目录的基本路径，该目录将包含我们的图像
BASE_PATH = "datasets/idc"

# 训练、验证和测试目录
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# 定义将用于训练的数据量
TRAIN_SPLIT = 0.8

# 验证数据量将是训练数据的百分比
VAL_SPLIT = 0.1