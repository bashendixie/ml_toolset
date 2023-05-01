# 加载 推理
import onnxruntime as ort
import torch
import time
import onnx

from PIL import Image
import cv2
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as alb
import torch.utils.data


# import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from albumentations import Compose
class test_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + '.png'))
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['images']

        img = img.astype('float32') / 255

        img = img.transpose(2, 1, 0)

        return img, {'img_id': img_id}


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def test_2():

    ort_session = ort.InferenceSession('model.onnx')
    input_name = ort_session.get_inputs()[0].name

    img = cv2.imread('inputs/data-science-bowl-2018/stage1_train/6bc8cda54f5b66a2a27d962ac219f8075bf7cc43b87ba0c9e776404370429e80/images/6bc8cda54f5b66a2a27d962ac219f8075bf7cc43b87ba0c9e776404370429e80.png')  # 02_test.tif')#demo.png
    #img = cv2.resize(img, (96, 96))

    nor = alb.Normalize()
    img = nor.apply(image=img)
    img = img.astype('float32') / 255
    #img = img.transpose(2, 1, 0)
    img = cv2.resize(img, (96, 96))

    tensor = transforms.ToTensor()(img)
    tensor = tensor.unsqueeze_(0)

    ort_outs = ort_session.run(None, {input_name: tensor.cpu().numpy()})

    img_out = ort_outs[0]
    img_out = torch.from_numpy(img_out)
    img_out = torch.sigmoid(img_out).cpu().numpy()

    cv2.imwrite(os.path.join('222222.png'), (img_out[0][0] * 255).astype('uint8'))



def test_1():
    ort_session = ort.InferenceSession('model.onnx')  # torch13.onnx')#'./semseg.onnx')
    onnx_input_name = ort_session.get_inputs()[0].name
    onnx_input_names = ort_session.get_inputs()
    onnx_outputs_names = ort_session.get_outputs()
    output_names = []
    for o in onnx_outputs_names:
        output_names.append(o.name)

    img = cv2.imread('inputs/data-science-bowl-2018/stage1_train/6bc8cda54f5b66a2a27d962ac219f8075bf7cc43b87ba0c9e776404370429e80/images/6bc8cda54f5b66a2a27d962ac219f8075bf7cc43b87ba0c9e776404370429e80.png')  # 02_test.tif')#demo.png
    img = cv2.resize(img, (96, 96))  # 256, 256)) # height = 1024, width = 2048   《2048,1024

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 4. onnxruntime上运行 onnx 模型；
    tensor = transforms.ToTensor()(img)
    # tensor=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)

    tensor = tensor.unsqueeze_(0)
    # tensor = tensor# / 255.0

    input_name = ort_session.get_inputs()[0].name
    label_name = ort_session.get_outputs()[0].name
    print("input_name ", input_name)
    print("label_name ", label_name)
    result = ort_session.run([label_name], {input_name: tensor.cpu().numpy()})
    # print("result ",result[0][0][0][303])#[303])


    # 输出保存semseg图
    # draw semseg mask images
    img2 = cv2.imread('inputs/data-science-bowl-2018/stage1_train/6bc8cda54f5b66a2a27d962ac219f8075bf7cc43b87ba0c9e776404370429e80/images/6bc8cda54f5b66a2a27d962ac219f8075bf7cc43b87ba0c9e776404370429e80.png', )  # 01_test.tif')#demo.png')
    img2 = cv2.resize(img2, (96, 96))  # 256,256)) # you can also use other way to create a temp images
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    for k in range(0, 1):
        for h in range(0, img2.shape[0]):
            for w in range(0, img2.shape[1]):
                img2[h, w] = result[0][0][k][h][w]

        # cv2.normalize(img2, img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)  # 归一化
        min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(img2)
        cv2.imwrite('./mask_semseg_' + str(k) + '.png', img2)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def test_n():
    """
    实际测试onnx模型效果
    :return:
    """
    onnx_path = 'model.onnx'
    image_path = 'inputs/data-science-bowl-2018/stage1_train'

    test_transform = Compose([
        transforms.Normalize(),
    ])

    test_dataset = test_Dataset(
        img_ids='0',
        img_dir=image_path,
        num_classes=1,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )
    # print(test_loader)
    for input, meta in test_loader:
        ort_session = ort.InferenceSession(onnx_path)
        # print('input', input.shape)
        # print(input.shape)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
        # print('ort_inputs', len(ort_inputs))
        ort_outs = ort_session.run(None, ort_inputs)
        # print('ort_outs', type(ort_outs))
        img_out = ort_outs[0]
        img_out = torch.from_numpy(img_out)
        # print('1', img_out)
        img_out = torch.sigmoid(img_out).cpu().numpy()

        # print('img_out', img_out.shape)
        img_out = img_out.transpose(0, 1, 3, 2)
        num_classes = 1
        for i in range(len(img_out)):
            cv2.imwrite(os.path.join('./', meta['img_id'][i].split('.')[0] + '.png'),
                        (img_out[i, num_classes - 1] * 255 * 255).astype('uint8'))



test_2()