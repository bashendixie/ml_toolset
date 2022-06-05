# 准备测试使用1280    runs 0.566\train\l6_3600_uflip_vm5_f132\weights
# 准备测试使用2160    runs 2160\train\l6_3600_uflip_vm5_f116\weights
# 混合进行判断
# 如果一方有但小于0.5，另一方没有则丢弃。
# 如果两方都有，都大于0.5时取。
# 如果两方都有，取分高的。

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import cv2
from tqdm import tqdm
import sys
from PIL import Image

model_1280 = torch.hub.load('D:/Project/kaggle/kaggle_torch_yolo5',
                       'custom',
                       path='runs 0.566/train/l6_3600_uflip_vm5_f132/weights/best.pt',
                       source='local',
                       force_reload=True)  # local repo
model_1280.conf = 0.01
#model_1280.iou

model_2160 = torch.hub.load('D:/Project/kaggle/kaggle_torch_yolo5',
                       'custom',
                       path='runs 2160/train/l6_3600_uflip_vm5_f116/weights/best.pt',
                       source='local',
                       force_reload=True)  # local repo
model_2160.conf = 0.01


trains = pd.read_csv('train.csv')

k=0
for item in trains.values:
    if item[5] != '[]':
        path = "img/" + str(item[0]) + "-" + str(item[2]) + ".jpg"
        path_1280 = "demo/1280/" + str(item[0]) + "-" + str(item[2]) + ".jpg"
        path_2160 = "demo/2160/" + str(item[0]) + "-" + str(item[2]) + ".jpg"

        img = Image.open(path)
        img_1280 = cv2.imread(path)
        img_2160 = cv2.imread(path)

        k = k+1
        print(k)

        anno_1280 = ''
        r_1280 = model_1280(img, size=1280, augment=True)
        if r_1280.pandas().xyxy[0].shape[0] == 0:
            anno_1280 = ''
            cv2.imwrite(path_1280, img_1280)
        else:
            for idx, row in r_1280.pandas().xyxy[0].iterrows():
                if row.confidence > 0.15:
                    cv2.rectangle(img_1280, (int(row.xmin), int(row.ymin)),(int(row.xmax), int(row.ymax)), (0, 0, 255), 2)
                    #cv2.putText(img_1280, str(row.confidence), )
                    cv2.imwrite(path_1280, img_1280)


        anno_2160 = ''
        r_2160 = model_2160(img, size=2160, augment=True)
        if r_2160.pandas().xyxy[0].shape[0] == 0:
            anno_2160 = ''
            cv2.imwrite(path_2160, img_2160)
        else:
            for idx, row in r_2160.pandas().xyxy[0].iterrows():
                if row.confidence > 0.15:
                    cv2.rectangle(img_2160, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)), (0, 0, 255), 2)
                    cv2.imwrite(path_2160, img_2160)









#
# img = Image.open('img/0-329.jpg')#cv2.imread('img/0-329.jpg')
#
# anno_1280 = ''
# r_1280 = model_1280(img, size=1280, augment=True)
# if r_1280.pandas().xyxy[0].shape[0] == 0:
#     anno_1280 = ''
# else:
#     for idx, row in r_1280.pandas().xyxy[0].iterrows():
#         if row.confidence > 0.28:
#             anno_1280 += '{} {} {} {} {} '.format(row.confidence, int(row.xmin), int(row.ymin), int(row.xmax-row.xmin), int(row.ymax-row.ymin))
#
#
# anno_2160 = ''
# r_2160 = model_2160(img, size=2160, augment=True)
# if r_2160.pandas().xyxy[0].shape[0] == 0:
#     anno_2160 = ''
# else:
#     for idx, row in r_2160.pandas().xyxy[0].iterrows():
#         if row.confidence > 0.28:
#             anno_2160 += '{} {} {} {} {} '.format(row.confidence, int(row.xmin), int(row.ymin), int(row.xmax-row.xmin), int(row.ymax-row.ymin))



