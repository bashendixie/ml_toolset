import os #it's a python library to import all the images that are saved in the folder named as dataSet
import cv2 # library
import numpy as np #opencv only works with numpy array
from PIL import Image #to capture images, PIL means python images library
recognizer = cv2.face.LBPHFaceRecognizer_create(); #create a recognizer, LBPH is a face recognition algorithm.Local Binary Patterns Histograms
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
path='dataSet'  #path of the images sample where the images are saved
def getImagesWithID(path): # a method to get all the corresponding images with id which means the saved images name
	imagePaths=[os.path.join(path,f) for f in os.listdir(path)] #create a list for the path for all the images that are available in the folder
	faces=[]
	IDs=[] # create empty list for the id
	for imagePath in imagePaths: #to go through the images
	# first open the images from directory than convert it to numpy array because opencv only works with numpy array
		faceImg=Image.open(imagePath).convert("L"); #faceImg is now PIL images, now opening the images from directory
		faceNp=np.array(faceImg,'uint8') #than converting to numpy array so opencv can work with it,uint8 meaning - Unsigned integer (0 to 255)
		fs = faceDetect.detectMultiScale(faceNp, 1.3, 5);
		for (x, y, w, h) in fs:
			#now need to get the user id which are name of the saved images in dataSet folder
			ID=int(os.path.split(imagePath)[-1].split('.')[0]) #to count backward of the saved images name, it is in string format that's why int is used to convert it to integer format
			faces.append(faceNp[y:y+h,x:x+w]) #directly store faces and id
			print(imagePath)
			print(ID)
			IDs.append(ID)
		cv2.imshow("training",faceNp)#images capturing or training
		cv2.waitKey(10)
	return np.array(IDs), faces #return the values means faces and ids
Ids, faces=getImagesWithID(path) # create faces,ids with path
recognizer.train(faces,Ids) # to train the recognizer, need the faces and ids
recognizer.save('recognizer/trainingData.yml') #create a recognizer folder in the present directory than got the trainingData.yml
cv2.destroyAllWindows()