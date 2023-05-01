# import the necessary packages
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False, help="base path for frozen checkpoint detection graph", default="D:/Project/deeplearn/pre_train_model/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")#model_export/
ap.add_argument("-l", "--labels", required=False, help="labels file", default="D:/Project/deeplearn/dataset/dlib_front_and_rear_vehicles_v1/records/classes.pbtxt")
ap.add_argument("-i", "--images", required=False, help="path to input images", default="C:/Users/zyh/Desktop/2.png")
ap.add_argument("-n", "--num-classes", type=int, required=False, help="# of class labels", default=2)
ap.add_argument("-c", "--min-confidence", type=float, default=0.75,help="minimum probability used to filter weak detections")
args = vars(ap.parse_args())

# initialize a set of colors for our class labels
COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 8))


loaded = tf.saved_model.load(args["model"])
#print(list(loaded.signatures.keys()))
infer = loaded.signatures["serving_default"]
image = cv2.imread(args["images"])
img = tf.expand_dims(image,axis=0)
result = infer(tf.constant(img))
print(result)

boxes = result["detection_boxes"]
scores = result["detection_scores"]
labels = result["detection_classes"]

# load the class labels from disk
labelMap = label_map_util.load_labelmap(args["labels"])
categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=args["num_classes"], use_display_name=True)
categoryIdx = label_map_util.create_category_index(categories)

boxes = np.squeeze(boxes)
scores = np.squeeze(scores)
labels = np.squeeze(labels)

output = image.copy()
# loop over the bounding box predictions
for (box, score, label) in zip(boxes, scores, labels):
    # if the predicted probability is less than the minimum
    # confidence, ignore it
    if score < args["min_confidence"]:
        continue
    # scale the bounding box from the range [0, 1] to [W, H]
    (startY, startX, endY, endX) = box
    startX = int(startX * 512)
    startY = int(startY * 512)
    endX = int(endX * 512)
    endY = int(endY * 512)
    # draw the prediction on the output images
    label = "ccc"#categoryIdx[masks]
    idx = 0#int(masks["id"]) - 1
    #masks = "{}: {:.2f}".format(masks["name"], score)
    cv2.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(output, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)#COLORS[idx]
# show the output images
cv2.imshow("Output", output)
cv2.waitKey(0)


#
# f = imported.signatures["serving_default"]
#
# image_np = np.load_image_into_numpy_array(args["images"])
# input_tensor = tf.convert_to_tensor(image_np)
#
#
# print(f(x=tf.constant([[1.]])))


