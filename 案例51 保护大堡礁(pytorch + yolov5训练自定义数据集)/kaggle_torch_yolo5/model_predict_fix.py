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

Image.LOAD_TRUNCATED_IMAGES = True

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



def computeIou(rec1,rec2):
    left_column_max = max(rec1.xmin, rec2.xmin)
    right_column_min = min(rec1.xmax, rec2.xmax)
    up_row_max = max(rec1.ymin, rec2.ymin)
    down_row_min = min(rec1.ymax, rec2.ymax)

    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
        # 两矩形有相交区域的情况
    else:
        S1 = (rec1.xmax-rec1.xmin)*(rec1.ymax-rec1.ymin)
        S2 = (rec2.xmax-rec2.xmin)*(rec2.ymax-rec2.ymin)
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)




trains = pd.read_csv('train.csv')

k=0
for item in trains.values:
    if item[5] != '[]':
        path = "img/" + str(item[0]) + "-" + str(item[2]) + ".jpg"
        path_1280 = "demo/2160first/" + str(item[0]) + "-" + str(item[2]) + ".jpg"
        path_2160 = "demo/2160/" + str(item[0]) + "-" + str(item[2]) + ".jpg"

        img = Image.open(path)
        img_1280 = cv2.imread(path)
        img_2160 = cv2.imread(path)

        k = k+1
        print(k)

        arr_1280 = []
        anno_1280 = ''
        r_1280 = model_1280(img, size=1280, augment=True)
        if r_1280.pandas().xyxy[0].shape[0] == 0:
            anno_1280 = ''
        else:
            for idx, row in r_1280.pandas().xyxy[0].iterrows():
                if row.confidence > 0.15:
                    arr_1280.append(row)
                    anno_1280 += '{} {} {} {} {} '.format(row.confidence, int(row.xmin), int(row.ymin), int(row.xmax-row.xmin), int(row.ymax-row.ymin))

        arr_2160 = []
        anno_2160 = ''
        r_2160 = model_2160(img, size=2160, augment=True)
        if r_2160.pandas().xyxy[0].shape[0] == 0:
            anno_2160 = ''
        else:
            for idx, row in r_2160.pandas().xyxy[0].iterrows():
                if row.confidence > 0.15:
                    arr_2160.append(row)
                    anno_2160 += '{} {} {} {} {} '.format(row.confidence, int(row.xmin), int(row.ymin), int(row.xmax-row.xmin), int(row.ymax-row.ymin))

        anno = ''
        if len(arr_1280)>0 and len(arr_2160)>0:
            temp_arr = []
            exist = False
            for arr1 in arr_2160:
                for arr2 in arr_1280:
                    # 如果相交
                    if computeIou(arr1, arr2)>0.7:
                        exist = True
                        if arr1.confidence > arr2.confidence:
                            temp_arr.append(arr1)
                        else:
                            temp_arr.append(arr2)
                if exist==False and arr1.confidence>0.7:
                    temp_arr.append(arr1)
        else:
           if len(arr_1280)==0 and len(arr_2160)==0:
               anno = ''
           else:
               if len(arr_1280) > 0 and len(arr_2160) == 0:
                    for arr in arr_1280:
                        anno += '{} {} {} {} {} '.format(arr.confidence, int(arr.xmin), int(arr.ymin),
                                                              int(arr.xmax - arr.xmin), int(arr.ymax - arr.ymin))
               else:
                   for arr in arr_2160:
                       anno += '{} {} {} {} {} '.format(arr.confidence, int(arr.xmin), int(arr.ymin),
                                                        int(arr.xmax - arr.xmin), int(arr.ymax - arr.ymin))

        for rec in temp_arr:
            cv2.rectangle(img_1280, (int(rec.xmin), int(rec.ymin)),(int(rec.xmax), int(rec.ymax)), (0, 0, 255), 2)
            cv2.imwrite(path_1280, img_1280)




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



