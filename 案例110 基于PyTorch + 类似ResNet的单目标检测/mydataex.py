import torch
import torchvision.transforms.functional as tf
import numpy as np

class Resize:
  '''Resize the image and convert the label
     to the new shape of the image'''
  def __init__(self, new_size=(256, 256)):
    self.new_width = new_size[0]
    self.new_height = new_size[1]

  def __call__(self, image_label_sample):
    image = image_label_sample[0]
    label = image_label_sample[1]
    c_x, c_y = label
    original_width, original_height = image.size
    image_new = tf.resize(image, (self.new_width, self.new_height))
    c_x_new = c_x * self.new_width /original_width
    c_y_new = c_y * self.new_height / original_height
    return image_new, (c_x_new, c_y_new)


class RandomHorizontalFlip:
  '''Horizontal flip the image with probability p.
     Adjust the label accordingly'''
  def __init__(self, p=0.5):
    if not 0 <= p <= 1:
      raise ValueError(f'Variable p is a probability, should be float between 0 to 1')
    self.p = p  # float between 0 to 1 represents the probability of flipping

  def __call__(self, image_label_sample):
    image = image_label_sample[0]
    label = image_label_sample[1]
    w, h = image.size
    c_x, c_y = label
    if np.random.random() < self.p:
      image = tf.hflip(image)
      label = w - c_x, c_y
    return image, label


class RandomVerticalFlip:
  '''Vertically flip the image with probability p.
    Adjust the label accordingly'''
  def __init__(self, p=0.5):
    if not 0 <= p <= 1:
      raise ValueError(f'Variable p is a probability, should be float between 0 to 1')
    self.p = p  # float between 0 to 1 represents the probability of flipping

  def __call__(self, image_label_sample):
    image = image_label_sample[0]
    label = image_label_sample[1]
    w, h = image.size
    c_x, c_y = label
    if np.random.random() < self.p:
      image = tf.vflip(image)
      label = c_x, h - c_y
    return image, label


class RandomTranslation:
  '''Translate the image by randomaly amount inside a range of values.
     Translate the label accordingly'''
  def __init__(self, max_translation=(0.2, 0.2)):
    if (not 0 <= max_translation[0] <= 1) or (not 0 <= max_translation[1] <= 1):
      raise ValueError(f'Variable max_translation should be float between 0 to 1')
    self.max_translation_x = max_translation[0]
    self.max_translation_y = max_translation[1]

  def __call__(self, image_label_sample):
    image = image_label_sample[0]
    label = image_label_sample[1]
    w, h = image.size
    c_x, c_y = label
    x_translate = int(np.random.uniform(-self.max_translation_x, self.max_translation_x) * w)
    y_translate = int(np.random.uniform(-self.max_translation_y, self.max_translation_y) * h)
    image = tf.affine(image, translate=(x_translate, y_translate), angle=0, scale=1, shear=0)
    label = c_x + x_translate, c_y + y_translate
    return image, label


class ImageAdjustment:
  '''Change the brightness and contrast of the image and apply Gamma correction.
     No need to change the label.'''
  def __init__(self, p=0.5, brightness_factor=0.8, contrast_factor=0.8, gamma_factor=0.4):
    if not 0 <= p <= 1:
      raise ValueError(f'Variable p is a probability, should be float between 0 to 1')
    self.p = p
    self.brightness_factor = brightness_factor
    self.contrast_factor = contrast_factor
    self.gamma_factor = gamma_factor

  def __call__(self, image_label_sample):
    image = image_label_sample[0]
    label = image_label_sample[1]

    if np.random.random() < self.p:
      brightness_factor = 1 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
      image = tf.adjust_brightness(image, brightness_factor)

    if np.random.random() < self.p:
      contrast_factor = 1 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
      image = tf.adjust_contrast(image, contrast_factor)

    if np.random.random() < self.p:
      gamma_factor = 1 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
      image = tf.adjust_gamma(image, gamma_factor)

    return image, label

class ToTensor:
  '''Convert the image to a Pytorch tensor with
     the channel as first dimenstion and values
     between 0 to 1. Also convert the label to tensor
     with values between 0 to 1'''
  def __init__(self, scale_label=True):
    self.scale_label = scale_label

  def __call__(self, image_label_sample):
    image = image_label_sample[0]
    label = image_label_sample[1]
    w, h = image.size
    c_x, c_y = label

    image = tf.to_tensor(image)

    if self.scale_label:
      label = c_x/w, c_y/h
    label = torch.tensor(label, dtype=torch.float32)

    return image, label


class ToPILImage:
  '''Convert a tensor image to PIL Image.
     Also convert the label to a tuple with
     values with the image units'''
  def __init__(self, unscale_label=True):
    self.unscale_label = unscale_label

  def __call__(self, image_label_sample):
    image = image_label_sample[0]
    label = image_label_sample[1].tolist()

    image = tf.to_pil_image(image)
    w, h = image.size

    if self.unscale_label:
      c_x, c_y = label
      label = c_x*w, c_y*h

    return image, label