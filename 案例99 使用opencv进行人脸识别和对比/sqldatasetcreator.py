import numpy as np 
import cv2,os
from PIL import Image
import pickle
import sqlite3
recognizer = cv2.face.LBPHFaceRecognizer_create(); #create a recognizer, LBPH is a face recognition algorithm.Local Binary Patterns Histograms 
recognizer.read("recognizer\\trainingData.yml")

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
path = 'dataSet'
def getProfile(id):
	conn = sqlite3.connect("FaceBase.db")
	cmd="SELECT * FROM People"
	cursor=conn.execute(cmd)
	profile=None
	for row in cursor:
		profile=row
	conn.close()
	return profile
cam = cv2.VideoCapture(0);
font = cv2.FONT_HERSHEY_SIMPLEX #5=font size
fontscale = 1
fontcolor = (255,255,255)
stroke = 2
profiles={}

while(True):
	ret, frame = cam.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces=faceDetect.detectMultiScale(gray,1.3,5);
	for(x,y,w,h) in faces:
		id, conf = recognizer.predict(gray[y:y+h,x:x+w])
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
		if(conf<50):
			if(id==1):
				id="Munmun"
			elif(id==2):
				id="Mysha"

		else:
			id="Unknown"
		profile=getProfile(id)
		if(profile!=None):
			cv2.putText(frame,"Name:" +str(profile[1]), (x,y+h+30), font, fontscale, fontcolor, stroke)#print the number or value of the prediction, str(id) means the text we want to print, (x,y+h) is for the text to be in the face and (x,y) is for the text on the upper of the rectangle
			cv2.putText(frame,"Age:" +str(profile[2]), (x,y+h+60), font, fontscale, fontcolor, stroke)
			cv2.putText(frame,"Gender:" +str(profile[3]), (x,y+h+90), font, fontscale, fontcolor, stroke)
			cv2.putText(frame,"Criminal Records:" +str(profile[4]), (x,y+h+120), font, fontscale, fontcolor, stroke)
			cv2.putText(frame,"Profession:" +str(profile[5]), (x,y+h+150), font, fontscale, fontcolor, stroke)
	cv2.imshow("frame",frame);
	#cv2.waitKey(1);
	#if(sampleNum>20):
	#	break
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break;
cam.release()
cv2.destroyAllWindows()