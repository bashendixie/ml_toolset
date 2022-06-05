import config
import matplotlib.pyplot as plt
import torchvision
import torch
# 加载PGAN预训练模型
model = torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub",
	"PGAN", model_name="celebAHQ-512", pretrained=True,
	useGPU=config.USE_GPU)

# 样本随机噪声向量
(noise, _) = model.buildNoiseData(config.NUM_IMAGES)

# 将采样的噪声向量通过预训练的生成器
with torch.no_grad():
	generatedImages = model.test(noise)

# 可视化生成的图像
grid = torchvision.utils.make_grid(
	generatedImages.clamp(min=-1, max=1), nrow=config.NUM_IMAGES, scale_each=True, normalize=True)
plt.figure(figsize = (20,20))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

# 保存生成的图像可视化
torchvision.utils.save_image(generatedImages.clamp(min=-1, max=1),
	config.SAVE_IMG_PATH, nrow=config.NUM_IMAGES, scale_each=True,
	normalize=True)