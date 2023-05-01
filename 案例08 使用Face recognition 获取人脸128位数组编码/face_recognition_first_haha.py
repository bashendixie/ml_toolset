# 使用Face recognition获取人脸128位

from imutils.video import VideoStream
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import imutils


def train():
    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = ['C:Users/zyh/Desktop/Watson.png','C:/Users/zyh/Desktop/Sherlock.png']
    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # loop over the images paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the images path
        print("[INFO] processing images {}/{}".format(i + 1, len(imagePaths)))
        name = 'Watson'
        if i==1:
            name = 'Sherlock'
        # load the input images and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input images
        boxes = face_recognition.face_locations(rgb, model='cnn')
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open('C:/Users/zyh/Desktop/encodings.pickle', "wb")
    f.write(pickle.dumps(data))
    f.close()

def reimage():
    # load the known faces and embeddings
    print("[INFO] loading encodings...")
    data = pickle.loads(open('C:/Users/zyh/Desktop/encodings.pickle', "rb").read())
    # load the input images and convert it from BGR to RGB
    image = cv2.imread('C:/Users/zyh/Desktop/v2.jpg')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input images, then compute the facial embeddings
    # for each face
    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb, model='cnn')
    encodings = face_recognition.face_encodings(rgb, boxes)
    # initialize the list of names for each face detected
    names = []
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input images to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the images
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
        # show the output images
        cv2.imshow("Image", image)
        cv2.waitKey(0)

def revideo1():
    # load the known faces and embeddings
    print("[INFO] loading encodings...")
    data = pickle.loads(open('C:/Users/zyh/Desktop/encodings.pickle', "rb").read())
    # initialize the video stream and pointer to output video file, then
    cap = cv2.VideoCapture('C:/Users/zyh/Desktop/123.mp4')

    # loop over frames from the video file stream
    while(cap.isOpened()):
        # grab the frame from the threaded video stream
        ret, frame = cap.read()

        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input images to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)
            # draw the predicted face name on the images
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

        # check to see if we are supposed to display the output frame to
        # the screen
        if 1 > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    # do a bit of cleanup
    cap.release()
    cv2.destroyAllWindows()


# 实时视频，但是需要gpu，即使用cpu+hog，也会卡顿
def revideo():
    # load the known faces and embeddings
    print("[INFO] loading encodings...")
    data = pickle.loads(open('C:/Users/zyh/Desktop/encodings.pickle', "rb").read())
    # initialize the video stream and pointer to output video file, then
    # allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    writer = None
    #time.sleep(2.0)

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()

        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input images to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)
            # draw the predicted face name on the images
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

        # 写入视频
        # # if the video writer is None *AND* we are supposed to write
        # # the output video to disk initialize the writer
        # if writer is None and args["output"] is not None:
        #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        #     writer = cv2.VideoWriter(args["output"], fourcc, 20,
        #                              (frame.shape[1], frame.shape[0]), True)
        # # if the writer is not None, write the frame with recognized
        # # faces to disk
        # if writer is not None:
        #     writer.write(frame)

        # check to see if we are supposed to display the output frame to
        # the screen
        if 1 > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    # check to see if the video writer point needs to be released
    if writer is not None:
        writer.release()

reimage()