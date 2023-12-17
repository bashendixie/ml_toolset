import numpy as np
import decord
from decord import cpu, gpu
import torch
import cv2


from gluoncv.torch.utils.model_utils import download
from gluoncv.torch.data.transforms.videotransforms import video_transforms, volume_transforms
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.model_zoo import get_model

# 基于decord库读取
# url = 'https://github.com/bryanyzhu/tiny-ucf101/raw/master/abseiling_k400.mp4'
# video_fname = download(url)
# vr = decord.VideoReader('abseiling_k400.mp4', ctx=cpu(0))
# frame_id_list = range(0, 64, 2)
# video_data = vr.get_batch(frame_id_list).asnumpy()

# 改成基于opencv读取
video = cv2.VideoCapture("huadong_up22.avi")    #
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
uccess, frame = video.read()
video_data = []

for i in range(7):
    video_data.append(frame)


crop_size = 224
short_side_size = 256
transform_fn = video_transforms.Compose([video_transforms.Resize(short_side_size, interpolation='bilinear'),
                                         video_transforms.CenterCrop(size=(crop_size, crop_size)),
                                         volume_transforms.ClipToTensor(),
                                         video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


clip_input = transform_fn(video_data)
print('Video data is downloaded and preprocessed.')


config_file = 'i3d_resnet50_v1_kinetics400.yaml'
cfg = get_cfg_defaults()
cfg.merge_from_file(config_file)
model = get_model(cfg)
model.eval()
print('%s model is successfully loaded.' % cfg.CONFIG.MODEL.NAME)


with torch.no_grad():
    pred = model(torch.unsqueeze(clip_input, dim=0)).numpy()
print('The input video clip is classified to be class %d' % (np.argmax(pred)))