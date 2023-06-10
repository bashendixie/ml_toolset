import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
from torch import optim
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from skimage.io import imread
from skimage.transform import resize
import os
import random
import cv2
import torch
import torch.nn


torch.manual_seed(42)
np.random.seed(42)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
train_size = 1.0
lr = 1e-3
weight_decay = 1e-6
batch_size = 32
epochs = 2



class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.act0 = nn.ReLU()
        self.bn0 = nn.BatchNorm2d(16)
        self.pool0 = nn.MaxPool2d(kernel_size=(2, 2))

        self.enc_conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.enc_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.enc_conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.bottleneck_conv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)

        self.upsample0 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv0 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1)
        self.dec_act0 = nn.ReLU()
        self.dec_bn0 = nn.BatchNorm2d(128)

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1)
        self.dec_act1 = nn.ReLU()
        self.dec_bn1 = nn.BatchNorm2d(64)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.dec_conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1)
        self.dec_act2 = nn.ReLU()
        self.dec_bn2 = nn.BatchNorm2d(32)

        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e0 = self.pool0(self.bn0(self.act0(self.enc_conv0(x))))
        e1 = self.pool1(self.bn1(self.act1(self.enc_conv1(e0))))
        e2 = self.pool2(self.bn2(self.act2(self.enc_conv2(e1))))
        e3 = self.pool3(self.bn3(self.act3(self.enc_conv3(e2))))

        b = self.bottleneck_conv(e3)

        d0 = self.dec_bn0(self.dec_act0(self.dec_conv0(self.upsample0(b))))
        d1 = self.dec_bn1(self.dec_act1(self.dec_conv1(self.upsample1(d0))))
        d2 = self.dec_bn2(self.dec_act2(self.dec_conv2(self.upsample2(d1))))
        d3 = self.sigmoid(self.dec_conv3(self.upsample3(d2)))
        return d3
#
# def getFileList(dir, Filelist, ext=None):
#     newDir = dir
#     if os.path.isfile(dir):
#         if ext is None:
#             Filelist.append(dir)
#         else:
#             if ext in dir[-3:]:
#                 Filelist.append(dir)
#
#     elif os.path.isdir(dir):
#         for s in os.listdir(dir):
#             newDir = os.path.join(dir, s)
#             getFileList(newDir, Filelist, ext)
#
#     return Filelist
#
# def load_dataset(train_part, root='PH2Dataset'):
#     images = []
#     masks = []
#
#     imagePaths = []
#     imagePaths = sorted(list(getFileList('C:\\Users\\zyh\\Desktop\\123', imagePaths)))
#     labelPaths = []
#     labelPaths = sorted(list(getFileList('C:\\Users\\zyh\\Desktop\\123', labelPaths)))
#
#     for imagePath in imagePaths:
#         images.append(imread(imagePath))
#     for labelPath in labelPaths:
#         masks.append(imread(labelPath))
#
#
#     size = (256, 256)
#     images = torch.FloatTensor(np.array([resize(image, size, mode='constant', anti_aliasing=True, ) for image in images])).unsqueeze(1)
#     masks = torch.FloatTensor(np.array([resize(mask, size, mode='constant', anti_aliasing=False) > 0.5 for mask in masks])).unsqueeze(1)
#
#     indices = np.random.permutation(range(len(images)))
#     train_part = int(train_part * len(images))
#     train_ind = indices[:train_part]
#     test_ind = indices[train_part:]
#
#     train_dataset = (images[train_ind, :], masks[train_ind, :])
#     test_dataset = (images[test_ind, :], masks[test_ind, :])
#
#     return train_dataset, test_dataset
#
# def plotn(n, data, only_mask=False):
#     images, masks = data[0], data[1]
#     fig, ax = plt.subplots(1, n)
#     fig1, ax1 = plt.subplots(1, n)
#     for i, (img, mask) in enumerate(zip(images, masks)):
#         if i == n:
#             break
#         if not only_mask:
#             ax[i].imshow(img)
#         else:
#             ax[i].imshow(img[0])
#         ax1[i].imshow(mask[0])
#         ax[i].axis('off')
#         ax1[i].axis('off')
#     plt.show()



# model = torch.load('model.pth', map_location=device)
# model.eval()
# input_names = ['input']
# output_names = ['output']
# x = torch.randn(1, 1, 256, 256, device=device)
# torch.onnx.export(model, x, 'model.onnx', input_names=input_names, output_names=output_names, verbose='True')
#





import torch.onnx
from torch.autograd import Variable
the_model = SegNet()
the_model.load_state_dict(torch.load('param_model.pth'))
the_model.eval()
dummy_input1 = torch.randn(1, 1, 256, 256)
# dummy_input2 = torch.randn(1, 3, 64, 64)
# dummy_input3 = torch.randn(1, 3, 64, 64)
# input_names = ["actual_input_1"]
# output_names = ["output1"]
# torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(the_model, dummy_input1, "model.onnx", verbose=True, opset_version = 11) #, input_names=input_names, output_names=output_names






# train_dataset, test_dataset = load_dataset(train_size)
#
# predictions = []
# image_mask = []
# plots = 5
# images, masks = train_dataset[0], train_dataset[1]
# for i, (img, mask) in enumerate(zip(images, masks)):
#     if i == plots:
#         break
#     img = img.to(device).unsqueeze(0)
#     predictions.append((model(img).detach().cpu()[0] > 0.5).float())
#     image_mask.append(mask)
# plotn(plots, (predictions, image_mask), only_mask=True)
