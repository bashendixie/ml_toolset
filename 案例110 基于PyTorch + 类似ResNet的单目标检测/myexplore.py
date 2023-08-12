from pathlib import Path
import pandas as pd

path_to_parent_dir = Path('.')
path_to_labels_file = 'Fovea_location.xlsx'

labels_df = pd.read_excel(path_to_labels_file, index_col='ID')

print('Head')
print(labels_df.head())  # show the first 5 rows in the excell file
print('\nTail')
print(labels_df.tail())  # show the last 5 rows in the excell file


import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 6)

amd_or_non_amd = ['AMD' if name.startswith('A') else 'Non-AMD' for name in labels_df.imgName]
sns.scatterplot(x='Fovea_X', y='Fovea_Y', hue=amd_or_non_amd, data=labels_df, alpha=0.7)
plt.show()


import numpy as np
from PIL import Image, ImageDraw

def load_image_with_label(labels_df, id):
  image_name = labels_df.loc[id, 'imgName']
  data_type = 'AMD' if image_name.startswith('A') else 'Non-AMD'
  image_path = path_to_parent_dir / 'Training400' / data_type / image_name
  image = Image.open(image_path)
  label = (labels_df.loc[id, 'Fovea_X'], labels_df.loc[id, 'Fovea_Y'])
  return image, label

def show_image_with_bounding_box(image, label, w_h_bbox=(50, 50), thickness=2):
  w, h = w_h_bbox
  c_x , c_y = label
  image = image.copy()
  ImageDraw.Draw(image).rectangle(((c_x-w//2, c_y-h//2), (c_x+w//2, c_y+h//2)), outline='green', width=thickness)
  plt.imshow(image)


rng = np.random.default_rng(42)  # create Generator object with seed 42
n_rows = 2  # number of rows in the image subplot
n_cols = 3  # # number of cols in the image subplot
indexes = rng.choice(labels_df.index, n_rows * n_cols)

for ii, id in enumerate(indexes, 1):
  image, label = load_image_with_label(labels_df, id)
  plt.subplot(n_rows, n_cols, ii)
  show_image_with_bounding_box(image, label, (250, 250), 20)
  plt.title(labels_df.loc[id, 'imgName'])

plt.show()