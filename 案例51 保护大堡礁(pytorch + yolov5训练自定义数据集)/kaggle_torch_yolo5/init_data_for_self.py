# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from shutil import copyfile

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

train = pd.read_csv('D:\\dataset\\s_labels.csv')
#train['pos'] = train.annotations != '[]'

fold = 1

annos = []
for i, x in train.iterrows():
    #copyfile(f's_images/{x.image_id}.jpg', f's_train/{x.image_id}.jpg')
    img = cv2.imread(f's_images/s_{x.image_id}.jpg')

    r = ''
    r_x = ''
    r_y = ''
    r_xy = ''
    anno = eval(x.labels)
    for an in anno:
        r += '0 {} {} {} {}\n'.format((an['x'] + an['width'] / 2) / img.shape[1],
                                      (an['y'] + an['height'] / 2) / img.shape[0],
                                      an['width'] / img.shape[1], an['height'] / img.shape[0])

        r_x += '0 {} {} {} {}\n'.format(1 - (an['x'] + an['width'] / 2) / img.shape[1],
                                        (an['y'] + an['height'] / 2) / img.shape[0],
                                        an['width'] / img.shape[1], an['height'] / img.shape[0])

        r_y += '0 {} {} {} {}\n'.format((an['x'] + an['width'] / 2) / img.shape[1],
                                        1 - (an['y'] + an['height'] / 2) / img.shape[0],
                                        an['width'] / img.shape[1], an['height'] / img.shape[0])

        r_xy += '0 {} {} {} {}\n'.format(1 - (an['x'] + an['width'] / 2) / img.shape[1],
                                         1 - (an['y'] + an['height'] / 2) / img.shape[0],
                                         an['width'] / img.shape[1], an['height'] / img.shape[0])

    with open(f's_label/origin/s_{x.image_id}.txt', 'w') as fp:
        fp.write(r)

    with open(f's_label/x/s_x_{x.image_id}.txt', 'w') as fp:
        fp.write(r_x)

    with open(f's_label/y/s_y_{x.image_id}.txt', 'w') as fp:
        fp.write(r_y)

    with open(f's_label/xy/s_xy_{x.image_id}.txt', 'w') as fp:
        fp.write(r_xy)

