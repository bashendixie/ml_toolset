import cv2
import sqlite3
cam = cv2.VideoCapture(0)
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

def insertOrUpdate(Id,Name):
	conn=sqlite3.connect("FaceBase.db")
	cmd="SELECT * FROM People"
	cursor=conn.execute(cmd)
	isRecordExist=0
	for row in cursor:
		isRecordExist=1
	if(isRecordExist==1):
		cmd="UPDATE People SET Name= "+str(Name)+" WHERE ID = "+str(Id)
	else:
		cmd="INSERT INTO PEOPLE(ID,Name) Values("+str(Id)+","+str(Name)+")"	 
	conn.execute(cmd)  
	conn.commit()
	conn.close()

id=input('Enter user id : ')
name=input('Enter your name : ')
insertOrUpdate(id, name)
sampleNum=0
while(True):
	ret, frame = cam.read();
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces=faceDetect.detectMultiScale(gray,1.3,5);
	for(x,y,w,h) in faces:
		sampleNum=sampleNum+1;
		cv2.imwrite("dataSet/user."+id+'.'+ str(sampleNum) +".jpg",frame[y:y+h,x:x+w])
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)	
	
	cv2.imshow('frame', frame) #imgshow,it is declare as frame that's why it doesn't shows gray color
	if cv2.waitKey(20) & 0xFF == ord('q'):#to close the frame just press q
		break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
