import numpy as np
import cv2,os
from PIL import Image
import pickle
import sqlite3
recognizer = cv2.face.LBPHFaceRecognizer_create(); #create a recognizer, LBPH is a face recognition algorithm.Local Binary Patterns Histograms
recognizer.read("recognizer\\trainingData.yml")
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
font = cv2.FONT_HERSHEY_SIMPLEX #5=font size
fontscale = 1
fontcolor = (255,255,255)
stroke = 2
profiles={}


frame = cv2.imread("n.png")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray,1.3,5);
for(x,y,w,h) in faces:
    id, conf = recognizer.predict(gray[y:y+h,x:x+w])
    print(id)
    print(conf)
    if conf<50:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frame,"ID:" +str(id), (x,y+h+30), font, fontscale, fontcolor, stroke)
cv2.imshow("frame",frame);
cv2.waitKey();