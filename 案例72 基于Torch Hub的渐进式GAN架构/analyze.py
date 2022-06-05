import config
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch


def interpolate(n):
    # 对两个噪声向量 z1 和 z2 进行采样
    (noise, _) = model.buildNoiseData(2)
    # 以步长间隔定义 (0, 1) 范围内的步长和样本数
    step = 1 / n
    lam = list(np.arange(0, 1, step))

    # 初始化用于存储插值图像的张量
    interpolatedImages = torch.zeros([n, 3, 512, 512])
    # 遍历 lam 的每个值
    for i in range(n):
        # 计算插值 z
        zInt = (1 - lam[i]) * noise[0] + lam[i] * noise[1]

        # 在图像空间中生成对应的
        with torch.no_grad():
            outputImage = model.test(zInt.reshape(-1, 512))
            interpolatedImages[i] = outputImage
    # 返回插值图像
    return interpolatedImages


# 加载预训练的 PGAN 模型
model = torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub",
                       "PGAN", model_name="celebAHQ-512", pretrained=True, useGPU=True)
# 调用插值函数
interpolatedImages = interpolate(config.NUM_INTERPOLATION)
# 可视化输出图像
grid = torchvision.utils.make_grid(
    interpolatedImages.clamp(min=-1, max=1), scale_each=True,
    normalize=True)
plt.figure(figsize=(20, 20))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
# 保存可视化
torchvision.utils.save_image(interpolatedImages.clamp(min=-1, max=1),
                             config.INTERPOLATE_PLOT_PATH, nrow=config.NUM_IMAGES,
                             scale_each=True, normalize=True)