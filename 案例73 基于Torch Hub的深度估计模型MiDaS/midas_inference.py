from data_utils import get_dataloader
import config
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torch


import os
# 使用测试转换管道创建测试数据集并初始化测试数据加载器
testTransform = Compose([
	Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), ToTensor()])
testDataset = ImageFolder(config.TEST_PATH, testTransform)
testLoader = get_dataloader(testDataset, config.PRED_BATCH_SIZE)

# 使用 torch hub 初始化 midas 模型
modelType = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", modelType)
# 将模型刷入设备并将其设置为评估模式
midas.to(config.DEVICE)
midas.eval()

# 初始化可迭代变量
sweeper = iter(testLoader)
# 抓取一批测试数据将图像发送到设备
print("[INFO] getting the test data...")
batch = next(sweeper)
(images, _) = (batch[0], batch[1])
images = images.to(config.DEVICE)

# 关闭自动毕业
with torch.no_grad():
	# 从输入中获取预测
	prediction = midas(images)
	# 批量解压预测
	prediction = torch.nn.functional.interpolate(
		prediction.unsqueeze(1), size=[384, 384], mode="bicubic",
		align_corners=False).squeeze()
# 将预测存储在一个 numpy 数组中
output = prediction.cpu().numpy()

# 定义行和列变量
rows = config.PRED_BATCH_SIZE
cols = 2
# 为子图定义轴
axes = []
fig=plt.figure(figsize=(10, 20))
# l遍历行和列
for totalRange in range(rows*cols):
	axes.append(fig.add_subplot(rows, cols, totalRange+1))
	# 为并排绘制基本事实和预测设置条件
	if totalRange % 2 == 0:
		plt.imshow(images[totalRange//2]
			.permute((1, 2, 0)).cpu().detach().numpy())
	else :
		plt.imshow(output[totalRange//2])
fig.tight_layout()
# 构建 midas 输出目录（如果尚未存在）
if not os.path.exists(config.MIDAS_OUTPUT):
	os.makedirs(config.MIDAS_OUTPUT)
# 将绘图保存到输出目录
print("[INFO] saving the inference...")
outputFileName = os.path.join(config.MIDAS_OUTPUT, "output.png")
plt.savefig(outputFileName)