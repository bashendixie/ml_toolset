import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import extcolors
import math

def zone(x, y):
    return 0.5 * (1 + math.cos(x * x + y * y))

SIZE = 597
#image = np.zeros((SIZE, SIZE))
image = cv2.imread('C:\\Users\\zyh\\Desktop\\111.bmp')

start = -8.2
end = 8.2
step = 0.0275

def dist_center(y, x):
    global SIZE
    center = SIZE / 2
    return math.sqrt( (x - center )**2 + (y - center )**2)

# for y in range(0, SIZE):
#     for x in range(0, SIZE):
#         if dist_center(y, x) > 300:
#             continue
#         y_val = start + y * step
#         x_val = start + x * step
#         image[y, x] = zone(x_val, y_val)

kernel_size = 15


def gkern(ksize):
    kernel_1d = cv2.getGaussianKernel(ksize=kernel_size, sigma=1, ktype=cv2.CV_32F)
    kernel_2d = kernel_1d * kernel_1d.T
    return kernel_2d

lowpass_kernel_gaussian = gkern(kernel_size)
lowpass_kernel_gaussian = lowpass_kernel_gaussian / lowpass_kernel_gaussian.sum()

#lowpass_kernel_gaussian = np.ones((kernel_size, kernel_size))
#lowpass_kernel_gaussian = lowpass_kernel_gaussian / lowpass_kernel_gaussian.sum()

lowpass_kernel_box = np.ones((kernel_size, kernel_size))
lowpass_kernel_box = lowpass_kernel_box / (kernel_size * kernel_size)

lowpass_image_gaussian = cv2.filter2D(image, -1, lowpass_kernel_gaussian)
lowpass_image_box = cv2.filter2D(image, -1, lowpass_kernel_box)


cv2.imwrite('C:\\Users\\zyh\\Desktop\\222.jpg', lowpass_image_gaussian)
cv2.imwrite('C:\\Users\\zyh\\Desktop\\333.jpg', lowpass_image_box)
cv2.imshow("o", image)
cv2.imshow("g", lowpass_image_gaussian)
cv2.imshow("b", lowpass_image_box)
cv2.waitKey(0)