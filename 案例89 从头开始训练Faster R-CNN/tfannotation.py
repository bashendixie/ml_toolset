# import the necessary packages
from object_detection.utils.dataset_util import bytes_list_feature
from object_detection.utils.dataset_util import float_list_feature
from object_detection.utils.dataset_util import int64_list_feature
from object_detection.utils.dataset_util import int64_feature
from object_detection.utils.dataset_util import bytes_feature


class TFAnnotation:
    def __init__(self):
        # initialize the bounding box + masks lists
        self.xMins = []
        self.xMaxs = []
        self.yMins = []
        self.yMaxs = []
        self.textLabels = []
        self.classes = []
        self.difficult = []

        # initialize additional variables, including the images
        # itself, spatial dimensions, encoding, and filename
        self.image = None
        self.width = None
        self.height = None
        self.encoding = None
        self.filename = None

    def build(self):
        # encode the attributes using their respective TensorFlow
        # encoding function
        w = int64_feature(self.width)
        h = int64_feature(self.height)
        filename = bytes_feature(self.filename.encode("utf8"))
        encoding = bytes_feature(self.encoding.encode("utf8"))
        image = bytes_feature(self.image)
        xMins = float_list_feature(self.xMins)
        xMaxs = float_list_feature(self.xMaxs)
        yMins = float_list_feature(self.yMins)
        yMaxs = float_list_feature(self.yMaxs)
        textLabels = bytes_list_feature(self.textLabels)
        classes = int64_list_feature(self.classes)
        difficult = int64_list_feature(self.difficult)
        # construct the TensorFlow-compatible data dictionary
        data = {
            "images/height": h,
            "images/width": w,
            "images/filename": filename,
            "images/source_id": filename,
            "images/encoded": image,
            "images/format": encoding,
            "images/object/bbox/xmin": xMins,
            "images/object/bbox/xmax": xMaxs,
            "images/object/bbox/ymin": yMins,
            "images/object/bbox/ymax": yMaxs,
            "images/object/class/text": textLabels,
            "images/object/class/masks": classes,
            "images/object/difficult": difficult,
        }
        # return the data dictionary
        return data