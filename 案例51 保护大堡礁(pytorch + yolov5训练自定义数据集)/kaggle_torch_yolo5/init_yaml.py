hyps = '''
# YOLOv5 by Ultralytics, GPL-3.0 license
# Hyperparameters for COCO training from scratch
# python train.py --batch 40 --cfg yolov5m.yaml --weights '' --data coco.yaml --img 640 --epochs 300
# See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.1  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # images HSV-Hue augmentation (fraction)
hsv_s: 0.7  # images HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # images HSV-Value augmentation (fraction)
degrees: 0.0  # images rotation (+/- deg)
translate: 0.1  # images translation (+/- fraction)
scale: 0.5  # images scale (+/- gain)
shear: 0.0  # images shear (+/- deg)
perspective: 0.0  # images perspective (+/- fraction), range 0-0.001
flipud: 0.5  # images flip up-down (probability)
fliplr: 0.5  # images flip left-right (probability)
mosaic: 1.0  # images mosaic (probability)
mixup: 0.5  # images mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)
'''


data = '''
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../yolo_data/fold1/  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/val  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 1  # number of classes
names: ['reef']  # class names


# Download script/URL (optional)
# download: https://ultralytics.com/assets/coco128.zip
'''

with open('data/reef_f1_naive.yaml', 'w') as fp:
    fp.write(data)
with open('data/hyps/hyp.heavy.2.yaml', 'w') as fp:
    fp.write(hyps)