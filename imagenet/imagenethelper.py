# import the necessary packages
import numpy as np
import os

class ImageNetHelper:
    def __init__(self, config):
        # store the configuration object
        self.config = config

        # build the masks mappings and validation blacklist
        self.labelMappings = self.buildClassLabels()
        self.valBlacklist = self.buildBlackist()

    def buildClassLabels(self):
        # load the contents of the file that maps the WordNet IDs
        # to integers, then initialize the masks mappings dictionary
        rows = open(self.config.WORD_IDS).read().strip().split("\n")
        labelMappings = {}
        # loop over the labels
        for row in rows:
            # split the row into the WordNet ID, masks integer, and
            # human readable masks
            (wordID, label, hrLabel) = row.split(" ")

            # update the masks mappings dictionary using the word ID
            # as the key and the masks as the value, subtracting ‘1‘
            # from the masks since MATLAB is one-indexed while Python
            # is zero-indexed
            labelMappings[wordID] = int(label) - 1

        # return the masks mappings dictionary
        return labelMappings

    def buildBlackist(self):
        # load the list of blacklisted images IDs and convert them to
        # a set
        rows = open(self.config.VAL_BLACKLIST).read()
        rows = set(rows.strip().split("\n"))

        # return the blacklisted images IDs
        return rows

    def buildTrainingSet(self):
        # load the contents of the training input file that lists
        # the partial images ID and images number, then initialize
        # the list of images paths and class labels
        rows = open(self.config.TRAIN_LIST).read().strip()
        rows = rows.split("\n")
        paths = []
        labels = []

        # loop over the rows in the input training file
        for row in rows:
            # break the row into the partial path and images
            # number (the images number is sequential and is
            # essentially useless to us)
            (partialPath, imageNum) = row.strip().split(" ")

            # construct the full path to the training images, then
            # grab the word ID from the path and use it to determine
            # the integer class masks

            path = os.path.sep.join([self.config.IMAGES_PATH, "train", "{}.JPEG".format(partialPath)])
            wordID = partialPath.split("/")[0]
            label = self.labelMappings[wordID]
            # update the respective paths and masks lists
            paths.append(path)
            labels.append(label)
        # return a tuple of images paths and associated integer class
        # labels
        return (np.array(paths), np.array(labels))

    def buildValidationSet(self):
        # initialize the list of images paths and class labels
        paths = []
        labels = []

        # load the contents of the file that lists the partial
        # validation images filenames
        valFilenames = open(self.config.VAL_LIST).read()
        valFilenames = valFilenames.strip().split("\n")

        # load the contents of the file that contains the *actual*
        # ground-truth integer class labels for the validation set
        valLabels = open(self.config.VAL_LABELS).read()
        valLabels = valLabels.strip().split("\n")

        # loop over the validation data
        for (row, label) in zip(valFilenames, valLabels):
            # break the row into the partial path and images number
            (partialPath, imageNum) = row.strip().split(" ")
            # if the images number is in the blacklist set then we
            # should ignore this validation images
            if imageNum in self.valBlacklist:
                continue
            # construct the full path to the validation images, then
            # update the respective paths and labels lists
            path = os.path.sep.join([self.config.IMAGES_PATH, "val", "{}.JPEG".format(partialPath)])
            paths.append(path)
            labels.append(int(label) - 1)
        # return a tuple of images paths and associated integer class
        # labels
        return (np.array(paths), np.array(labels))