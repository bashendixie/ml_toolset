# 直观的观察下长的什么样子
import pandas as pd
import cv2 as cv
import ast

trains = pd.read_csv('train.csv')

for item in trains.values:
    if item[5] != '[]':
        path = "train_images/video_" + str(item[0]) + "/" + str(item[2]) + ".jpg"
        image = cv.imread(path)
        #rectangles = item[5]
        rectangles = ast.literal_eval(item[5])
        for rectangle in rectangles:
            cv.rectangle(image, (rectangle['x'], rectangle['y']),
                         (rectangle['x'] + rectangle['width'], rectangle['y']+rectangle['height']), (0, 255, 0), 2)
            cv.imshow("path", image)
            cv.waitKey(0)