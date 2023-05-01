# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker images: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from shutil import copyfile

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

train = pd.read_csv('train.csv')
train['pos'] = train.annotations != '[]'

fold = 1

annos = []
for i, x in train.iterrows():
    if x.video_id == fold:
        mode = 'train'
        if not x.pos: continue
    else:
        # train
        mode = 'train'
        if not x.pos: continue
        # val
    copyfile(f'train_images/video_{x.video_id}/{x.video_frame}.jpg',
                f'jiubi1/fold{fold}/images/{mode}/{x.image_id}.jpg')
    if not x.pos:
        continue
    r = ''
    r_x = ''
    r_y = ''
    r_xy = ''
    anno = eval(x.annotations)
    for an in anno:
        r += '0 {} {} {} {}\n'.format((an['x'] + an['width'] / 2) / 1280,
                                        (an['y'] + an['height'] / 2) / 720,
                                        an['width'] / 1280, an['height'] / 720)

        r_x += '0 {} {} {} {}\n'.format(1 - (an['x'] + an['width'] / 2) / 1280,
                                      (an['y'] + an['height'] / 2) / 720,
                                      an['width'] / 1280, an['height'] / 720)

        r_y += '0 {} {} {} {}\n'.format((an['x'] + an['width'] / 2) / 1280,
                                      1 - (an['y'] + an['height'] / 2) / 720,
                                      an['width'] / 1280, an['height'] / 720)

        r_xy += '0 {} {} {} {}\n'.format(1 - (an['x'] + an['width'] / 2) / 1280,
                                      1 - (an['y'] + an['height'] / 2) / 720,
                                      an['width'] / 1280, an['height'] / 720)

    with open(f'jiubi1/fold{fold}/labels/{mode}/{x.image_id}.txt', 'w') as fp:
        fp.write(r)

    #with open(f'yolo_data/fold{fold}/x_labels/{x.image_id}.txt', 'w') as fp:
    #    fp.write(r_x)

    #with open(f'yolo_data/fold{fold}/y_labels/{x.image_id}.txt', 'w') as fp:
    #    fp.write(r_y)

    #with open(f'yolo_data/fold{fold}/xy_labels/{x.image_id}.txt', 'w') as fp:
    #    fp.write(r_xy)
