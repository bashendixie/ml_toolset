# import the necessary packages
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_segmentation_masks(allClassMask, images, numClasses,
                                 inverseTransforms, device):
    # 转换为布尔掩码和批量维度第一格式
    booleanMask = (allClassMask == torch.arange(numClasses, device=device)[:, None, None, None])
    booleanMask = booleanMask.transpose(1, 0)
    # 初始化列表以存储我们的输出掩码
    outputMasks = []

    # 遍历所有图像和相应的布尔掩码
    for image, mask in zip(images, booleanMask):
        # 在输入图像上绘制分割掩码
        outputMasks.append(
            draw_segmentation_masks(
                (inverseTransforms(image) * 255).to(torch.uint8),
                masks=mask,
                alpha=0.6
            )
        )
    # 返回分割图
    return outputMasks