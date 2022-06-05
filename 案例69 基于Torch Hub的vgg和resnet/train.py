# 运行命令
# python train.py --model vgg
# python train.py --model resnet
import config
from classifier import Classifier
from datautils import get_dataloader
from datautils import train_val_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomRotation
from torchvision.transforms import Normalize
from torch.nn import CrossEntropyLoss
from torch.nn import Softmax
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import torch
# 构造参数解析器并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="vgg", choices=["vgg", "resnet"], help="name of the backbone model")
args = vars(ap.parse_args())

# 检查主干模型的名称是否为 VGG
if args["model"] == "vgg":
	# load VGG-11 model
	baseModel = torch.hub.load("pytorch/vision:v0.10.0", "vgg11", pretrained=True, skip_validation=True)
	# 冻结 VGG-11 模型的层
	for param in baseModel.features.parameters():
		param.requires_grad = False
# 否则，我们将使用的主干模型是 ResNet
elif args["model"] == "resnet":
    # load ResNet 18 model
    baseModel = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True, skip_validation=True)

    # 定义模型的最后一层和当前层
    lastLayer = 8
    currentLayer = 1
    # 循环遍历模型的子层
    for child in baseModel.children():
        # 检查我们是否还没有到达最后一层
        if currentLayer < lastLayer:
            # 循环子层的参数并冻结它们
            for param in child.parameters():
                param.requires_grad = False
        # 否则，我们已经到达最后一层，所以打破循环
        else:
            break
        # 增加当前层
        currentLayer += 1

# 定义转换管道
trainTransform = Compose([
	RandomResizedCrop(config.IMAGE_SIZE),
	RandomHorizontalFlip(),
	RandomRotation(90),
	ToTensor(),
	Normalize(mean=config.MEAN, std=config.STD)
])
# 使用 ImageFolder 创建训练数据集
trainDataset = ImageFolder(config.TRAIN_PATH, trainTransform)

# 创建训练和验证数据拆分
(trainDataset, valDataset) = train_val_split(dataset=trainDataset)
# 创建训练和验证数据加载器
trainLoader = get_dataloader(trainDataset, config.BATCH_SIZE)
valLoader = get_dataloader(valDataset, config.BATCH_SIZE)

# 构建自定义模型
model = Classifier(baseModel=baseModel.to(config.DEVICE), numClasses=2, model=args["model"])
model = model.to(config.DEVICE)
# 初始化损失函数和优化器
lossFunc = CrossEntropyLoss()
lossFunc.to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LR)
# 初始化 softmax 激活层
softmax = Softmax()

# 计算训练和验证集的每轮步数
trainSteps = len(trainDataset) // config.BATCH_SIZE
valSteps = len(valDataset) // config.BATCH_SIZE
# 初始化一个字典来存储训练历史
H = {
	"trainLoss": [],
	"trainAcc": [],
	"valLoss": [],
	"valAcc": []
}

# 循环遍历
print("[INFO] training the network...")
for epoch in range(config.EPOCHS):
    # 将模型设置为训练模式
    model.train()

    # 初始化总训练和验证损失
    totalTrainLoss = 0
    totalValLoss = 0

    # 初始化训练和验证步骤中正确预测的数量
    trainCorrect = 0
    valCorrect = 0

    # 循环训练集
    for (image, target) in tqdm(trainLoader):
        # send the input to the device
        (image, target) = (image.to(config.DEVICE), target.to(config.DEVICE))

        # 执行前向传递并计算训练损失
        logits = model(image)
        loss = lossFunc(logits, target)
        # 将梯度归零，执行反向传播步骤，并更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 将损失添加到到目前为止的总训练损失中，将输出 logits 通过 softmax 层得到输出预测，并计算正确预测的数量
        totalTrainLoss += loss.item()
        pred = softmax(logits)
        trainCorrect += (pred.argmax(dim=-1) == target).sum().item()

    # 关闭自动
    with torch.no_grad():
        # 将模型设置为评估模式
        model.eval()

        # 循环验证集
        for (image, target) in tqdm(valLoader):
            # 将输入发送到设备
            (image, target) = (image.to(config.DEVICE), target.to(config.DEVICE))
            # 做出预测并计算验证损失
            logits = model(image)
            valLoss = lossFunc(logits, target)
            totalValLoss += valLoss.item()

            # 将输出logits通过softmax层得到输出预测，并计算正确预测的数量
            pred = softmax(logits)
            valCorrect += (pred.argmax(dim=-1) == target).sum().item()

    # 计算平均训练和验证损失
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    # 计算训练和验证准确度
    trainCorrect = trainCorrect / len(trainDataset)
    valCorrect = valCorrect / len(valDataset)
    # 更新我们的培训历史
    H["trainLoss"].append(avgTrainLoss)
    H["valLoss"].append(avgValLoss)
    H["trainAcc"].append(trainCorrect)
    H["valAcc"].append(valCorrect)
    # 打印模型训练和验证信息
    print(f"[INFO] EPOCH: {epoch + 1}/{config.EPOCHS}")
    print(f"Train loss: {avgTrainLoss:.6f}, Train accuracy: {trainCorrect:.4f}")
    print(f"Val loss: {avgValLoss:.6f}, Val accuracy: {valCorrect:.4f}")

# 绘制训练损失和准确率
plt.style.use("ggplot")
plt.figure()
plt.plot(H["trainLoss"], label="train_loss")
plt.plot(H["valLoss"], label="val_loss")
plt.plot(H["trainAcc"], label="train_acc")
plt.plot(H["valAcc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)

# 将模型状态序列化到磁盘
torch.save(model.state_dict(), config.MODEL_PATH)