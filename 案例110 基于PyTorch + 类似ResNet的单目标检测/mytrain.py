import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose
from mydataex import *
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.nn as nn
# from torchinfo import summary
from torchvision.ops import box_iou
import matplotlib.pyplot as plt

# 读取
path_to_parent_dir = Path('.')
path_to_labels_file = 'Fovea_location.xlsx'
labels_df = pd.read_excel(path_to_labels_file, index_col='ID')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AMDDataset(Dataset):
  def __init__(self, data_path, labels_df, transformation):
    self.data_path = Path(data_path)
    self.labels_df = labels_df.reset_index(drop=True)
    self.transformation = transformation

  def __getitem__(self, index):
    image_name = self.labels_df.loc[index, 'imgName']
    image_path = self.data_path / ('AMD' if image_name.startswith('A') else 'Non-AMD') / image_name
    image = Image.open(image_path)
    label = self.labels_df.loc[index, ['Fovea_X','Fovea_Y']].values.astype(float)
    image, label = self.transformation((image, label))
    return image.to(device), label.to(device)

  def __len__(self):
    return len(self.labels_df)


class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.base1 = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same'),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(True)
    )
    self.base2 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    x = self.base1(x) + x
    x = self.base2(x)
    return x


class FoveaNet(nn.Module):
  def __init__(self, in_channels, first_output_channels):
    super().__init__()
    self.model = nn.Sequential(
      ResBlock(in_channels, first_output_channels),
      nn.MaxPool2d(2),
      ResBlock(first_output_channels, 2 * first_output_channels),
      nn.MaxPool2d(2),
      ResBlock(2 * first_output_channels, 4 * first_output_channels),
      nn.MaxPool2d(2),
      ResBlock(4 * first_output_channels, 8 * first_output_channels),
      nn.MaxPool2d(2),
      nn.Conv2d(8 * first_output_channels, 16 * first_output_channels, kernel_size=3),
      nn.MaxPool2d(2),
      nn.Flatten(),
      nn.Linear(7 * 7 * 16 * first_output_channels, 2)
    )

  def forward(self, x):
    return self.model(x)


def centroid_to_bbox(centroids, w=50/256, h=50/256):
  x0_y0 = centroids - torch.tensor([w/2, h/2]).to(device)
  x1_y1 = centroids + torch.tensor([w/2, h/2]).to(device)
  return torch.cat([x0_y0, x1_y1], dim=1)


def iou_batch(output_labels, target_labels):
  output_bbox = centroid_to_bbox(output_labels)
  target_bbox = centroid_to_bbox(target_labels)
  return torch.trace(box_iou(output_bbox, target_bbox)).item()


def batch_loss(loss_func, output, target, optimizer=None):
  loss = loss_func(output, target)
  with torch.no_grad():
    iou_metric = iou_batch(output, target)
  if optimizer is not None:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return loss.item(), iou_metric


def train_val_step(dataloader, model, loss_func, optimizer=None):
  if optimizer is not None:
    model.train()
  else:
    model.eval()

  running_loss = 0
  running_iou = 0

  for image_batch, label_batch in dataloader:
    output_labels = model(image_batch)
    loss_value, iou_metric_value = batch_loss(loss_func, output_labels, label_batch, optimizer)
    running_loss += loss_value
    running_iou += iou_metric_value

  return running_loss / len(dataloader.dataset), running_iou / len(dataloader.dataset)

# 划分数据集
labels_df_train, labels_df_val = train_test_split(labels_df, test_size=0.2, shuffle=True, random_state=42)

# 进行数据转换
train_transformation = Compose(
  [Resize(), RandomHorizontalFlip(), RandomVerticalFlip(), RandomTranslation(), ImageAdjustment(), ToTensor()])
val_transformation = Compose([Resize(), ToTensor()])

# 生成dataloader
train_dataset = AMDDataset('Training400', labels_df_train, train_transformation)
val_dataset = AMDDataset('Training400', labels_df_val, val_transformation)
train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=16)


# 声明模型并训练
net = FoveaNet(3, 16)

# 打印模型结构，需要安装torchinfo
# summary(model=net,
#         input_size=(8, 3, 256, 256), # (batch_size, color_channels, height, width)
#         col_names=["input_size", "output_size", "num_params"],
#         col_width=20,
#         row_settings=["var_names"]
# )

loss_func = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

num_epoch = 100
loss_tracking = {'train': [], 'val': []}
iou_tracking = {'train': [], 'val': []}
best_loss = float('inf')

model = FoveaNet(3, 16).to(device)
loss_func = nn.SmoothL1Loss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(num_epoch):
  print(f'Epoch {epoch + 1}/{num_epoch}')

  training_loss, trainig_iou = train_val_step(train_dataloader, model, loss_func, optimizer)
  loss_tracking['train'].append(training_loss)
  iou_tracking['train'].append(trainig_iou)

  with torch.inference_mode():
    val_loss, val_iou = train_val_step(val_dataloader, model, loss_func, None)
    loss_tracking['val'].append(val_loss)
    iou_tracking['val'].append(val_iou)
    if val_loss < best_loss:
      print('Saving best model')
      torch.save(model.state_dict(), 'best_model.pt')
      best_loss = val_loss

  print(f'Training loss: {training_loss:.6}, IoU: {trainig_iou:.2}')
  print(f'Validation loss: {val_loss:.6}, IoU: {val_iou:.2}')


plt.plot(range(1, num_epoch+1), loss_tracking['train'], label='train')
plt.plot(range(1, num_epoch+1), loss_tracking['val'], label='validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()


plt.plot(range(1, num_epoch+1), iou_tracking['train'], label='train')
plt.plot(range(1, num_epoch+1), iou_tracking['val'], label='validation')
plt.xlabel('epoch')
plt.ylabel('iou')
plt.legend()



# 验证模型
def show_image_with_2_bounding_box(image, label, target_label, w_h_bbox=(50, 50), thickness=2):
  w, h = w_h_bbox
  c_x , c_y = label
  c_x_target , c_y_target = target_label
  image = image.copy()
  ImageDraw.Draw(image).rectangle(((c_x-w//2, c_y-h//2), (c_x+w//2, c_y+h//2)), outline='green', width=thickness)
  ImageDraw.Draw(image).rectangle(((c_x_target-w//2, c_y_target-h//2), (c_x_target+w//2, c_y_target+h//2)), outline='red', width=thickness)
  plt.imshow(image)


model.load_state_dict(torch.load('best_model.pt'))
model.eval()
rng = np.random.default_rng(0)  # create Generator object with seed 0
n_rows = 2  # number of rows in the image subplot
n_cols = 3  # # number of cols in the image subplot
indexes = rng.choice(range(len(val_dataset)), n_rows * n_cols, replace=False)

for ii, id in enumerate(indexes, 1):
  image, label = val_dataset[id]
  output = model(image.unsqueeze(0))
  iou = iou_batch(output, label.unsqueeze(0))
  _, label = ToPILImage()((image, label))
  image, output = ToPILImage()((image, output.squeeze()))
  plt.subplot(n_rows, n_cols, ii)
  show_image_with_2_bounding_box(image, output, label)
  plt.title(f'{iou:.2f}')