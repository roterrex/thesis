#from PlanGen.util import util
from util import util
import tensorflow as tf
import os
import pathlib
import time
import datetime
import yaml

from matplotlib import pyplot as plt

from testTrain import testTrain
from pixToPix import pix2Pix
#from IPython import display

conf = yaml.load(open("PlanGen/settings.yaml", 'r'), Loader=yaml.Loader)

dataset_name = "facades"

_URL = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'

path_to_zip = tf.keras.utils.get_file(
    fname=f"{dataset_name}.tar.gz",
    origin=_URL,
    extract=True)

path_to_zip  = pathlib.Path(path_to_zip)

PATH = path_to_zip.parent/dataset_name
print(PATH)

print(list(PATH.parent.iterdir()))

sample_image = tf.io.read_file(str(PATH / 'train/1.jpg'))
sample_image = tf.io.decode_jpeg(sample_image)
print(sample_image.shape)
plt.figure()
plt.imshow(sample_image)
plt.show()

def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)

  # Split each image tensor into two tensors:
  # - one with a real building facade image
  # - one with an architecture label image 
  w = tf.shape(image)[1]
  w = w // 2
  input_image = image[:, w:, :]
  real_image = image[:, :w, :]

  # Convert both images to float32 tensors
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = util.random_jitter(input_image, real_image, conf)
  input_image, real_image = util.normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = util.resize(input_image, real_image, 
                                        conf['LayoutGan']['ImgSize'])
  input_image, real_image = util.normalize(input_image, real_image)

  return input_image, real_image


train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg'))
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(conf['LayoutGan']['BufferSize'])
train_dataset = train_dataset.batch(conf['LayoutGan']['BatchSize'])

try:
  test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
except tf.errors.InvalidArgumentError:
  test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(conf['LayoutGan']['BatchSize'])


p2p = pix2Pix(conf) 


tt = testTrain(p2p, None, conf)

tt.fit(train_dataset, test_dataset, steps=5000)


