from detecto import utils
import matplotlib.pyplot as plt
import matplotlib.image as img
from torchvision import transforms
from detecto import core
from detecto import visualize

utils.xml_to_csv('train_labels', 'train.csv')
utils.xml_to_csv('val_labels', 'val.csv')

image = img.imread('images/n02085620_8611.jpg')
plt.imshow(image)
plt.show()


transform_img = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(800),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    utils.normalize_transform(),
])

dataset = core.Dataset('train.csv', 'images/', transform=transform_img)

image, information = dataset[50]
visualize.show_labeled_image(image, information['boxes'], information['labels'])


dataloader = core.DataLoader(dataset)
validation_data = core.Dataset('val.csv', 'images/')
categories = ['Chihuahua', 'golden_retriever']
classifier = core.Model(categories)
history = classifier.fit(dataloader, validation_data, epochs = 20, verbose = True)
plt.plot(history)

images = []
for i in range(0,36,3):
  image,_ = validation_data[i]
  images.append(image)

visualize.plot_prediction_grid(classifier, images, dim=(4, 3), figsize=(16, 12))