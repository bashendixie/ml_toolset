import config
from imutils import paths
import random
import shutil
import os

# 获取原始输入目录中所有输入图像的路径并将它们打乱
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)

# 计算训练和测试分割
i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# 我们将使用部分训练数据进行验证
i = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# 定义我们将要构建的数据集
datasets = [
    ("training", trainPaths, config.TRAIN_PATH),
    ("validation", valPaths, config.VAL_PATH),
    ("testing", testPaths, config.TEST_PATH)
]

# 循环数据集
for (dType, imagePaths, baseOutput) in datasets:
    # 打印我们正在创建的数据拆分
    print("[INFO] building '{}' split".format(dType))
    # 如果输出基本输出目录不存在，则创建它
    if not os.path.exists(baseOutput):
        print("[INFO] 'creating {}' directory".format(baseOutput))
        os.makedirs(baseOutput)
    # 循环输入图像路径
    for inputPath in imagePaths:
        # 提取输入图像的文件名并提取类标签（“0”表示“负”，“1”表示“正”）
        filename = inputPath.split(os.path.sep)[-1]
        label = filename[-5:-4]
        # 构建标签目录的路径
        labelPath = os.path.sep.join([baseOutput, label])
        # 如果标签输出目录不存在，则创建它
        if not os.path.exists(labelPath):
            print("[INFO] 'creating {}' directory".format(labelPath))
            os.makedirs(labelPath)
        # 构建目标图像的路径，然后复制图像本身
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)