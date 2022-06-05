# 直观的观察下长的什么样子
import pandas as pd
import cv2 as cv
import ast

trains = pd.read_csv('train.csv')

k=0
for item in trains.values:
    if item[5] != '[]':
        path = "demo/fix_copy/" + str(item[0]) + "-" + str(item[2]) + ".jpg"
        path_origin = "img/" + str(item[0]) + "-" + str(item[2]) + ".jpg"

        print(k)
        image = cv.imread(path)
        if image is None:
            image = cv.imread(path_origin)

        #rectangles = item[5]
        rectangles = ast.literal_eval(item[5])
        k = k+1
        for rectangle in rectangles:
            cv.rectangle(image, (rectangle['x'], rectangle['y']),
                         (rectangle['x'] + rectangle['width'], rectangle['y']+rectangle['height']), (0, 255, 0), 2)
            cv.imwrite(path, image)
            #cv.waitKey(0)