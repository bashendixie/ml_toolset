import config
import utils
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os

# 创建图像变换和逆变换
imageTransforms = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)])
imageInverseTransforms = transforms.Normalize(
	mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
	std=[1/0.229, 1/0.224, 1/0.225]
)
# 初始化数据集和数据加载器
print("[INFO] creating data pipeline...")
testDs = ImageFolder(config.TEST_PATH, imageTransforms)
testLoader = DataLoader(testDs, shuffle=True, batch_size=config.BATCH_SIZE)
# 加载预训练的 FCN 分割模型，将模型刷入设备，并将其设置为评估模式
print("[INFO] loading FCN segmentation model from Torch Hub...")
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load("pytorch/vision:v0.10.0", config.MODEL[0], pretrained=True)
model.to(config.DEVICE)
model.eval()
# 初始化迭代器并从数据集中抓取一批
batchIter = iter(testLoader)
print("[INFO] getting the test data...")
batch = next(batchIter)
# 解压图像和标签并移动到设备
(images, labels) = (batch[0], batch[1])
images = images.to(config.DEVICE)
# 初始化一个空列表来存储图像
imageList =[]
# 循环遍历所有图像
for image in images:
	# 将非规范化图像添加到列表中
	imageList.append(
		(imageInverseTransforms(image) * 255).to(torch.uint8)
	)

# 如果不存在则创建输出目录
if not os.path.exists(config.BASE_OUTPUT):
	os.makedirs(config.BASE_OUTPUT)
# turn off auto grad
with torch.no_grad():
	# 从模型计算预测
	output = model(images)["out"]
# 将预测转换为类别概率
normalizedMasks = torch.nn.functional.softmax(output, dim=1)
# 转换为像素级的类级分割掩码
classMask = normalizedMasks.argmax(1)
# 可视化分割掩码
outputMasks = utils.visualize_segmentation_masks(
	allClassMask=classMask,
	images=images,
	numClasses=config.NUM_CLASSES,
	inverseTransforms=imageInverseTransforms,
	device=config.DEVICE
)
# 将输入图像和输出掩码转换为张量
inputImages = torch.stack(imageList)
generatedMasks = torch.stack(outputMasks)
# 保存输入图像可视化和掩码可视化
print("[INFO] saving the image and mask visualization to disk...")
save_image(inputImages.float() / 255,
	config.SAVE_IMAGE_PATH, nrow=4, scale_each=True,
	normalize=True)
save_image(generatedMasks.float() / 255,
	config.SEGMENTATION_OUTPUT_PATH, nrow=4, scale_each=True,
	normalize=True)