# Face-detection-and-identification-with-mysql-database-in-real-time

Face Detection and identification by using mysql database and implementing opencv, python:

Face Detection: Face detection is just pulling out faces in an image or a video. 

Face Identification: Face identification is identify faces in a video or image including the person's name. This takes a little bit extra training, that we have done in here. The training portion and the identification of faces can be absolutely advanced. Advanced means using deep learning libraries such as tensorflow or PI torch. Opencv has some built-in features to make those things(advanced) easier. 

To do this project here used: numpy==1.15.4, opencv==3.4.3.18, pillow=5.3.0, virualenv==16.1.0, python=3.6.7, SQLiteStudio.

✔ Step 1: Download SQLiteStudio 

SQLiteStudio needs to be downloaded that suits your operating system from this link sqlitestudio.pl/index.rvt?act=download and install it. After insatalling open it and create a database named as FaceBase.db in a directory. In this directoery other file for this project will be here. In this db file now create a table which is here named as "People". Now, in this People table add some colum as per ur wish. I created Id, Name, Age, Female, Criminal Records by supposing it is a police record.

✔ Step 2: Copy haarcascade_frontalface_default.xml file 

Copy this file from cv2 or download it from this repository and keep this file in the project directory.

✔ Step 3: Create two directories dataSet and recognizer

Create new two directories in the project directory. I named one as dataSet and other one named as recognizer. In the dataSet directory pictures of the person will be saved automatically and in the recognizer directory a yml file will be saved automatically when the pictures will be trained.

✔ Step 4: Create a new py file for gathering data(file named as datasetgenerating.py)

This py file has been created in the same project directory. In this datasetgenerating.py file user id and user name has been asked. So, when running this datasetgenerating.py file enter user id and enter user name. The user name needs to be entered in double quotation. Than the webcamera will open and detect a face and take picture of the face and will save the pictures of faces in dataSet directory. 

✔ Step 5: Close the webcam by hit on q button of the keyboard

When the webcam will open and take pictures of face automatically than need to close the webcam by hitting q button not the ✖ button. By hitting ✖ button of the webcam, the webcam will not close. Than look at the FaceBase.db file in the people table and refresh it and will see the name and the id which has been given by the user.

✔ Step 6: Create another py file for train those pictures of faces(file named as facetrain.py)

This py file will train all the pictures of faces which are in the dataSet directory. After running facetrain.py file trainingData.yml will be automatically saved in recognizer directory. This will help to recognize a person.

✔ Step 7: Create one new another py file(file named as sqldatasetcreator.py)

In this file the person id and name has been declared in the if else loop. The same id and name has to be given which has been given as user input while running datasetgenerating.py file. Otherwise result will not be proper. Now, after running sqldatasetcreator.py file will see the name of the person and there also need to do some other works that is: in the FaceBase.db file's people table the other column need to be update. In the people table age, gender and criminal records will be written and save it. Than we can see in the webcam the person's name, age, gender, criminal records. We can also add another column in the people table if we want.  
