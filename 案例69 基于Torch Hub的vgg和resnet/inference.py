# USAGE
# python inference.py --model vgg
# python inference.py --model resnet
import config
from classifier import Classifier
from datautils import get_dataloader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from torchvision import transforms
from torch.nn import Softmax
from torch import nn
import matplotlib.pyplot as plt
import argparse
import torch
from tqdm import tqdm
# 构造参数解析器并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="vgg", choices=["vgg", "resnet"], help="name of the backbone model")
args = vars(ap.parse_args())

# 初始化测试转换管道
testTransform = Compose([
	Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	ToTensor(),
	Normalize(mean=config.MEAN, std=config.STD)
])
# 计算逆均值和标准差
invMean = [-m/s for (m, s) in zip(config.MEAN, config.STD)]
invStd = [1/s for s in config.STD]
# 定义我们的非规范化变换
deNormalize = transforms.Normalize(mean=invMean, std=invStd)
# 创建测试数据集
testDataset = ImageFolder(config.TEST_PATH, testTransform)
# 初始化测试数据加载器
testLoader = get_dataloader(testDataset, config.PRED_BATCH_SIZE)

# 检查主干模型的名称是否为 VGG
if args["model"] == "vgg":
	# load VGG-11 model
	baseModel = torch.hub.load("pytorch/vision:v0.10.0", "vgg11", pretrained=True, skip_validation=True)
# 否则，我们将使用的主干模型是 ResNet
elif args["model"] == "resnet":
	# load ResNet 18 model
	baseModel = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True, skip_validation=True)

# 构建自定义模型
model = Classifier(baseModel=baseModel.to(config.DEVICE), numClasses=2, model=args["model"])
model = model.to(config.DEVICE)
# 加载模型状态并初始化损失函数
model.load_state_dict(torch.load(config.MODEL_PATH))
lossFunc = nn.CrossEntropyLoss()
lossFunc.to(config.DEVICE)
# 初始化测试数据丢失
testCorrect = 0
totalTestLoss = 0
soft = Softmax()

# 关闭自动毕业
with torch.no_grad():
	# 将模型设置为评估模式
	model.eval()
	# 循环验证集
	for (image, target) in tqdm(testLoader):
		# 将输入发送到设备
		(image, target) = (image.to(config.DEVICE), target.to(config.DEVICE))
		# 做出预测并计算验证损失
		logit = model(image)
		loss = lossFunc(logit, target)
		totalTestLoss += loss.item()
		# 通过softmax层输出logits得到输出预测，并计算正确预测的个数
		pred = soft(logit)
		testCorrect += (pred.argmax(dim=-1) == target).sum().item()

# 打印测试数据准确性
print(testCorrect/len(testDataset))
# 初始化可迭代变量
sweeper = iter(testLoader)
# 抓取一批测试数据
batch = next(sweeper)
(images, labels) = (batch[0], batch[1])
# 初始化一个图形
fig = plt.figure("Results", figsize=(10, 10 ))

# 关闭自动毕业
with torch.no_grad():
	# 将图像发送到设备
	images = images.to(config.DEVICE)
	# 做出预测
	preds = model(images)
	# 循环所有批次
	for i in range(0, config.PRED_BATCH_SIZE):
		# 初始化子图
		ax = plt.subplot(config.PRED_BATCH_SIZE, 1, i + 1)
		# 抓取图像，对其进行去规范化，将原始像素强度缩放到 [0, 255] 范围内，并从通道前 tp 通道最后更改通道顺序
		image = images[i]
		image = deNormalize(image).cpu().numpy()
		image = (image * 255).astype("uint8")
		image = image.transpose((1, 2, 0))
		# 获取真实标签
		idx = labels[i].cpu().numpy()
		gtLabel = testDataset.classes[idx]
		# 获取预测标签
		pred = preds[i].argmax().cpu().numpy()
		predLabel = testDataset.classes[pred]
		# 将结果和图像添加到绘图中
		info = "Ground Truth: {}, Predicted: {}".format(gtLabel, predLabel)
		plt.imshow(image)
		plt.title(info)
		plt.axis("off")

	# 显示图标
	plt.tight_layout()
	plt.show()