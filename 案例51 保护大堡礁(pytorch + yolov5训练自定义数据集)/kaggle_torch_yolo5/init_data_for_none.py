# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from shutil import copyfile

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

train = pd.read_csv('train.csv')
train['pos'] = train.annotations != '[]'

fold = 1
max = 0
annos = []
for i, x in train.iterrows():
    if not x.pos:
        fold = 2
        #copyfile(f'train_images/video_{x.video_id}/{x.video_frame}.jpg', f'yolo_data/fold{fold}/nolabel/{x.image_id}.jpg')
    else:
        anno = eval(x.annotations)
        for an in anno:
            if max < an['width']:
                max = an['width']
            if max < an['height']:
                max = an['height']
        #continue
print(str(max))
