# import the necessary packages
import argparse
import time
import cv2
import os


model_path = "d:\\EDSR_x4.pb"

# extract the model name and model scale from the file path
modelName = model_path.split(os.path.sep)[-1].split("_")[0].lower()
modelScale = model_path.split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

# initialize OpenCV's super resolution DNN object, load the super
# resolution model from disk, and set the model name and scale
print("[INFO] loading super resolution model: {}".format(model_path))
print("[INFO] model name: {}".format(modelName))
print("[INFO] model scale: {}".format(modelScale))
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(model_path)
sr.setModel(modelName, modelScale)

# load the input images from disk and display its spatial dimensions
image = cv2.imread("C:\\Users\\zyh\\Desktop\\s_images\\20.jpg")
print("[INFO] w: {}, h: {}".format(image.shape[1], image.shape[0]))
# use the super resolution model to upscale the images, timing how
# long it takes
start = time.time()
upscaled = sr.upsample(image)
end = time.time()
print("[INFO] super resolution took {:.6f} seconds".format(end - start))
# show the spatial dimensions of the super resolution images
print("[INFO] w: {}, h: {}".format(upscaled.shape[1], upscaled.shape[0]))

# resize the images using standard bicubic interpolation
# start = time.time()
# bicubic = cv2.resize(images, (upscaled.shape[1], upscaled.shape[0]), interpolation=cv2.INTER_CUBIC)
# end = time.time()
# print("[INFO] bicubic interpolation took {:.6f} seconds".format(end - start))

# show the original input images, bicubic interpolation images, and
# super resolution deep learning output

#cv2.imshow("Original", images)
#cv2.imshow("Bicubic", bicubic)
#cv2.imshow("Super Resolution", upscaled)
cv2.imwrite("C:\\Users\\zyh\\Desktop\\20.jpg", upscaled)
cv2.waitKey(0)