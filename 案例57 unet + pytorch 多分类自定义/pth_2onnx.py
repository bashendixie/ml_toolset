import os
import cv2
import onnxruntime
import torch
from albumentations import Compose
from albumentations.augmentations import transforms
from torch.utils.data import DataLoader
import losses
import argparse
# import L1_archs_cut
# from dataset import test_Dataset
from config import UNetConfig
from unet import NestedUNet

def pth_2onnx():
    """
    pytorch 模型转换为onnx模型
    :return:
    """
    torch_model = torch.load('./data/checkpoints/epoch_10.pth ')
    cfg = UNetConfig()
    model = eval(cfg.model)(cfg)

    model.load_state_dict(torch_model)
    batch_size = 1  # 批处理大小
    input_shape = (3, 128, 128)  # 输入数据

    # set the model to inference mode
    model.eval()
    print(model)
    x = torch.randn(batch_size, *input_shape)  # 生成张量
    export_onnx_file = "model.onnx"  # 目的ONNX文件名
    torch.onnx.export(model,
                      x,
                      export_onnx_file,
                      # 注意这个地方版本选择为11
                      opset_version=11,
                      )
                      #   ,
                      # do_constant_folding=True,  # 是否执行常量折叠优化
                      # input_names=["input"],  # 输入名
                      # output_names=["output"],  # 输出名
                      # dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                      #               "output": {0: "batch_size"}})


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#
# def image_test():
#     """
#     实际测试onnx模型效果
#     :return:
#     """
#     onnx_path = 'model.onnx'
#     image_path = './data/ray_test/'
#
#     test_transform = Compose([
#         transforms.Normalize(),
#     ])
#
#     test_dataset = test_Dataset(
#         img_ids='0',
#         img_dir=image_path,
#         num_classes=1,
#         transform=test_transform
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=1,
#         drop_last=False
#     )
#     # print(test_loader)
#     for input, meta in test_loader:
#         ort_session = onnxruntime.InferenceSession(onnx_path)
#         # print('input', input.shape)
#         # print(input.shape)
#         ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
#         # print('ort_inputs', len(ort_inputs))
#         ort_outs = ort_session.run(None, ort_inputs)
#         # print('ort_outs', type(ort_outs))
#         img_out = ort_outs[0]
#         img_out = torch.from_numpy(img_out)
#         # print('1', img_out)
#         img_out = torch.sigmoid(img_out).cpu().numpy()
#
#         # print('img_out', img_out.shape)
#         img_out = img_out.transpose(0, 1, 3, 2)
#         num_classes = 1
#         for i in range(len(img_out)):
#             cv2.imwrite(os.path.join('./', meta['img_id'][i].split('.')[0] + '.png'),
#                         (img_out[i, num_classes - 1] * 255).astype('uint8'))


if __name__ == '__main__':
    pth_2onnx()
    # image_test()

