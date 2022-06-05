from PIL import ImageFilter as ImageFilter
from PIL import Image

import os

rootdir = 'D:/Project/kaggle/kaggle_torch_yolo5/yolo_sharp/fold1/images/val_o'
newdir = 'D:/Project/kaggle/kaggle_torch_yolo5/yolo_sharp/fold1/images/val/'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0,len(list)):
    path = os.path.join(rootdir, list[i])
    if os.path.isfile(path):
        im = Image.open(path)
        test_filter = ImageFilter.UnsharpMask(3.0, 200, 0)
        i = im.filter(test_filter)
        i.save(newdir + os.path.basename(path))