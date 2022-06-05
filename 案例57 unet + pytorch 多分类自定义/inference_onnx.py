# 加载 推理
import onnxruntime as ort
import torch
import time
import onnx
import onnx.onnx_operators_pb
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

    img = cv2.imread('./data/test/input/1582708143552.png')  # 02_test.tif')#demo.png
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_nd = np.array(img)
    if len(img_nd.shape) == 2:
        # mask target image
        img_nd = np.expand_dims(img_nd, axis=2)
    else:
        # grayscale input image
        # scale between 0 and 1
        img_nd = img_nd / 255

    img = img_nd.transpose(2, 0, 1)
    img = img.astype(float)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device='cpu', dtype=torch.float32)

    input_name = ort_session.get_inputs()[0].name
    label_name = ort_session.get_outputs()[0].name
    print("input_name ", input_name)
    print("label_name ", label_name)
    result = ort_session.run([label_name], {input_name: img.cpu().numpy()})

    # img_out = result[0]
    # img_out = torch.from_numpy(img_out)
    # img_out = torch.sigmoid(img_out).cpu().numpy()

    for k in range(0, 3):
        # img2 = cv2.imread('./data/test/input/1582708143405.png')
        # tensor = np.array(result[0][0][k])
        # tensor = np.expand_dims(tensor, axis=2)
        # tensor = transforms.ToTensor()(tensor)
        # #tensor = torch.sigmoid(tensor)
        # #tensor = tensor > 0.5
        # tensor = tensor.squeeze(0)
        # tensor = np.array(tensor)
        # img2 = tensor * 255

        # for h in range(0, tensor.shape[0]):
        #     for w in range(0, tensor.shape[1]):
        #         if tensor[h, w] > 0:
        #             img2[h, w] = 255
        #         else:
        #             img2[h, w] = 0

        img = result[0][0][k]
        cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
        print(img)
        #img = img>0
        img = img > 0.5
        cv2.imwrite('./mask_semseg_' + str(k) + '.png',  (img * 255).astype(np.uint8))


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()



test_1()