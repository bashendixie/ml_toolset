import os
import os.path as osp

import cv2 as cv
import numpy as np
from random import shuffle
from PIL import Image # 如果自己写代码的话可以不用, 用 OpenCV 就可以了

import matplotlib.pyplot as plt

# 取得 data_set 目录中的文件
# data_set_path: 数据集所在文件夹, 可以是文件夹列表, 因为你有可能将不同类别数据放到不同文件中
# split_rate: 这些文件中用于训练, 验证, 测试所占的比例
#             如果为 None, 则不区分, 直接返回全部
#             如果只写一个小数, 如 0.8, 则表示 80% 为训练集, 20% 为验证集, 没有测试集
#             如果是一个 tuple 或 list, 只有一个元素的话, 同上面的一个小数的情况
# shuffle_enable: 是否要打乱顺序
# 返回训练集, 验证集和验证集路径列表
def get_data_path(data_set_path, split_rate = (0.7, 0.2, 0.1), shuffle_enable = True):
    data_path = []
    
    if not isinstance(data_set_path, list):	# 变成列表可以以统一的方式进行操作
        data_set_path = [data_set_path]
   
    for dsi in data_set_path:
        folders = os.listdir(dsi)
        
        for folder in folders:
            if osp.isdir(osp.join(dsi, folder)):
                data_path.append(osp.join(dsi, folder))
                
    if shuffle_enable:
        shuffle(data_path)
        
    if None == split_rate:
        return data_path

    total_num = len(data_path)

    if isinstance(split_rate, float) or 1 == len(split_rate):
        if isinstance(split_rate, float):
            split_rate = [split_rate]
        train_pos = int(total_num * split_rate[0])
        train_set = data_path[: train_pos]
        valid_set = data_path[train_pos: ]

        return train_set, valid_set

    elif isinstance(split_rate, tuple) or isinstance(split_rate, list):
        list_len = len(split_rate)
        assert(list_len > 1)

        train_pos = int(total_num * split_rate[0])
        valid_pos = int(total_num * (split_rate[0] + split_rate[1]))

        train_set = data_path[0: train_pos]
        valid_set = data_path[train_pos: valid_pos]
        test_set = data_path[valid_pos: ]

        return train_set, valid_set, test_set


# 读图像和标签
# data_path: 就是上面 get_data_path 返回的路径
# batch_size: 一次加载图像的数量, -1 表示加载全部
# zero_based: True, 四周扩展, False, 右边和底边扩展
# train_mode: 训练模式, 对应的是测试模式, 测试模式会返回 roi 矩形, 方便还原原始的尺寸
# shuffle_enable: 是否要打乱数据, 这个是上面 get_data_path 打乱有什么不一样呢, 这个打乱是每一个 epoch 会
#                 会打乱一次
def segment_reader(data_path, batch_size, zero_based = False, train_mode = True, shuffle_enable = True):
    if not train_mode:
        if osp.isdir(data_path):
            bkup = data_path
            test_files = os.listdir(data_path)
            
            data_path = []
            for f in test_files:
                data_path.append(osp.join(bkup, f))
        else:
            data_path = [data_path]
            
    data_nums = len(data_path)
    index_list = [x for x in range(data_nums)]
    
    stop_now = False
    
    while False == stop_now:
        if shuffle_enable:
            shuffle(index_list)
            
        train = []
        label = []
        
        for i in index_list:
            read_path = data_path[i]
            if train_mode:
                read_path = osp.join(data_path[i], "img.png")
                
            if not osp.exists(read_path):
                continue
                
            img_src = cv.imread(read_path)
                
            shape = img_src.shape
            
            max_rows = max(64, shape[0])
            max_rows = max_rows // 32 * 32 + 32 # +32 是为了扩展边缘, 不然边缘的像素可能分割不好
            max_cols = max(64, shape[1])
            max_cols = max_cols // 32 * 32 + 32

            b = max_rows - shape[0]
            r = max_cols - shape[1]

            half_padding_x = 0
            half_padding_y = 0
            
            if False == zero_based:
                half_padding_x = r >> 1
                half_padding_y = b >> 1

            # 扩展边界
            big_x = cv.copyMakeBorder(img_src,
                                      half_padding_y, b - half_padding_y,
                                      half_padding_x, r - half_padding_x,
                                      cv.BORDER_REPLICATE, (0, 0, 0))

            # 转换成 0~1 的范围
            big_x = np.array(big_x).astype(np.float32) / 255.0

            if train_mode:                
                read_path = osp.join(data_path[i], "label.png")
                
                # 如果标签图像像素值就是类别的话, 可以这样读
                '''
                img_label = cv.imread(read_path, cv.IMREAD_GRAYSCALE)
                big_y = cv.copyMakeBorder(img_label,
                                          half_padding_y, b - half_padding_y,
                                          half_padding_x, r - half_padding_x,
                                          cv.BORDER_REPLICATE, (0, 0, 0))
                
                big_y = np.array(big_y).astype(np.float32) # 注意这里不用除以 255
                
                '''
                # 这里如果标签图像是索引图像的话, 可以这样读                
                img_label = np.array(Image.open(read_path, 'r'))
                big_y = cv.copyMakeBorder(img_label,
                                          half_padding_y, b - half_padding_y,
                                          half_padding_x, r - half_padding_x,
                                          cv.BORDER_REPLICATE, (0, 0, 0))
                big_y = big_y.astype(np.float32) # 注意这里不用除以 255
                
               
                train.append(big_x)
                label.append(big_y)
            else:
                roi = (half_padding_x, half_padding_y, shape[1], shape[0])

                train.append(big_x)
                label.append(roi)
            

            if len(train) == batch_size:
                if train_mode:
                    yield (np.array(train), np.array(label))
                else:
                    yield (np.array(train), label)

                train = []
                label = []
                
        if train:
            if train_mode:
                yield (np.array(train), np.array(label))
            else:
                yield (np.array(train), label)
            
            train = []
            label = []
            
        if False == train_mode:
            stop_now = True
