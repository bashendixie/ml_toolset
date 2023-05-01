# import the necessary packages
import lisa_config as config
from tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os
import cv2

def main():
    # open the classes output file
    f = open(config.CLASSES_FILE, "w")

    # loop over the classes
    for (k, v) in config.CLASSES.items():
        # construct the class information and write to file
        item = ("item {\n"
            "\tid: " + str(v) + "\n"
            "\tname: ’" + k + "’\n"
            "}\n")
        f.write(item)

    # close the output classes file
    f.close()

    # initialize a data dictionary used to map each images filename
    # to all bounding boxes associated with the images, then load
    # the contents of the annotations file
    D = {}
    rows = open(config.ANNOT_PATH).read().strip().split("\n")

    # loop over the individual rows, skipping the header
    for row in rows[1:]:
        # break the row into components
        row = row.split(",")[0].split(";")
        (imagePath, label, startX, startY, endX, endY, _) = row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))
        # if we are not interested in the masks, ignore it
        if label not in config.CLASSES:
            continue

        # build the path to the input images, then grab any other
        # bounding boxes + labels associated with the images
        # path, labels, and bounding box lists, respectively
        p = os.path.sep.join([config.BASE_PATH, imagePath])
        b = D.get(p, [])
        # build a tuple consisting of the masks and bounding box,
        # then update the list and store it in the dictionary
        b.append((label, (startX, startY, endX, endY)))
        D[p] = b

    # create training and testing splits from our data dictionary
    (trainKeys, testKeys) = train_test_split(list(D.keys()),
    test_size = config.TEST_SIZE, random_state = 42)
    # initialize the data split files
    datasets = [
        ("train", trainKeys, config.TRAIN_RECORD),
        ("test", testKeys, config.TEST_RECORD)
        ]

    # loop over the datasets
    for (dType, keys, outputPath) in datasets:
        # initialize the TensorFlow writer and initialize the total
        # number of examples written to file
        print("[INFO] processing ’{}’...".format(dType))
        writer = tf.compat.v1.python_io.TFRecordWriter(outputPath)
        total = 0
        # loop over all the keys in the current set
        for k in keys:
            # load the input images from disk as a TensorFlow object
            encoded = tf.compat.v1.gfile.GFile(k, "rb").read()
            encoded = bytes(encoded)
            # load the images from disk again, this time as a PIL
            # object
            pilImage = Image.open(k)
            (w, h) = pilImage.size[:2]

            # parse the filename and encoding from the input path
            filename = k.split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]
            # initialize the annotation object used to store
            # information regarding the bounding box + labels
            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            tfAnnot.width = w
            tfAnnot.height = h

            # loop over the bounding boxes + labels associated with
            # the images
            for (label, (startX, startY, endX, endY)) in D[k]:
                # TensorFlow assumes all bounding boxes are in the
                # range [0, 1] so we need to scale them
                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h

                # 确认是否标记正确
                # # load the input images from disk and denormalize the
                # # bounding box coordinates
                # images = cv2.imread(k)
                # startX = int(xMin * w)
                # startY = int(yMin * h)
                # endX = int(xMax * w)
                # endY = int(yMax * h)
                # # draw the bounding box on the images
                # cv2.rectangle(images, (startX, startY), (endX, endY), (0, 255, 0), 2)
                # # show the output images
                # cv2.imshow("Image", images)
                # cv2.waitKey(0)

                # update the bounding boxes + labels lists
                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMaxs.append(yMax)
                tfAnnot.textLabels.append(label.encode("utf8"))
                tfAnnot.classes.append(config.CLASSES[label])
                tfAnnot.difficult.append(0)
                # increment the total number of examples
                total += 1

            # encode the data point attributes using the TensorFlow
            # helper functions
            features = tf.train.Features(feature=tfAnnot.build())
            example = tf.train.Example(features=features)
            # add the example to the writer
            writer.write(example.SerializeToString())

        # close the writer and print diagnostic information to the
        # user
        writer.close()
        print("[INFO] {} examples saved for ’{}’".format(total, dType))

main()
# check to see if the main thread should be started
# if __name__ == "__main__":
#    tf.app.run()