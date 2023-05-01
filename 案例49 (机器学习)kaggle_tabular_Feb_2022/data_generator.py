import os
from PIL import Image
import numpy as np

BATCHSIZE = 10
root_path = '/home/eric/data/NUS-WIDE/images'


class data_generator:

    def __init__(self, file_path, _max_example, image_size, classes):
        self.load_data(file_path=file_path)
        self.index = 0
        self.batch_size = BATCHSIZE
        self.image_size = image_size
        self.classes = classes
        self.load_images_labels(_max_example)
        self.num_of_examples = _max_example

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            self.datasets = f.readlines()

    def load_images_labels(self, _max_example):
        images = []
        labels = []
        for i in range(0, len(self.datasets[:_max_example])):
            data_arr = self.datasets[i].strip().split('*')
            image_path = os.path.join(root_path, data_arr[0]).replace("\\", "/")
            img = Image.open(image_path)
            img = img.resize((self.image_size[0], self.image_size[1]), Image.ANTIALIAS)
            img = np.array(img)
            images.append(img)
            tags = data_arr[1].split(' ')
            label = np.zeros((self.classes))
            for i in range(1, len(tags)):
                #         print(word_id[tags[i]])
                id = int(word_id[tags[i]])
                label[id] = 1
            labels.append(label)
        self.images = images
        self.labels = labels

    def get_mini_batch(self):
        while True:
            batch_images = []
            batch_labels = []
            for i in range(self.batch_size):
                if (self.index == len(self.images)):
                    self.index = 0
                batch_images.append(self.images[self.index])
                batch_labels.append(self.labels[self.index])
                self.index += 1
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            yield batch_images, batch_labels


id_tag_path = 'word_id.txt'
word_id = {}
with open(id_tag_path, 'r') as f:
    words = f.readlines()
    for item in words:
        arr = item.strip().split(' ')
        word_id[arr[1]] = arr[0]

# if __name__ == "__main__":
#     txt_path = 'datasets81_clean.txt'
#     width, height = 224, 224
#     IMAGE_SIZE = (width, height, 3)
#     classes = 81
#     train_gen = data_generator(txt_path, 100, IMAGE_SIZE, classes)
#     x, y = next(train_gen.get_mini_batch())
#     print(x.shape)
#     print(y.shape)