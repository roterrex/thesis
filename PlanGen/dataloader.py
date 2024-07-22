import tensorflow as tf
from matplotlib import pyplot as plt

from util import util

class dataloader:
    

    def __init__(self, conf, ds_path) -> None:
        self.conf = conf
        self.ds_path = ds_path

    def plot_sample_image(self, run_plot): #conf['images']['plotSample']
        if run_plot:
            sample_image = tf.io.read_file(str(self.ds_path / 'train/1.jpg'))
            sample_image = tf.io.decode_jpeg(sample_image)
            print(sample_image.shape)
            plt.figure()
            plt.imshow(sample_image)
            plt.show()

    def load_ds(self):
        train_ds = tf.data.Dataset.list_files(str(self.ds_path / 'train/*.jpg'))
        train_ds = train_ds.map(self.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(self.conf['LayoutGan']['BufferSize'])
        train_ds = train_ds.batch(self.conf['LayoutGan']['BatchSize'])

        test_ds = tf.data.Dataset.list_files(str(self.ds_path / 'test/*.jpg'))
        test_ds = test_ds.map(self.load_image_test)
        test_ds = test_ds.batch(self.conf['LayoutGan']['BatchSize'])

        val_ds = tf.data.Dataset.list_files(str(self.ds_path / 'val/*.jpg'))
        val_ds = val_ds.map(self.load_image_test)
        val_ds = val_ds.batch(self.conf['LayoutGan']['BatchSize'])

        return train_ds, test_ds, val_ds      

    def read_to_tensor(self, image_file):
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
    
    def load_image_train(self, image_file):
        input_image, real_image = self.read_to_tensor(image_file)
        input_image, real_image = util.random_jitter(input_image, real_image, self.conf)
        input_image, real_image = util.normalize(input_image, real_image)

        return input_image, real_image
    
    def load_image_test(self, image_file):
        input_image, real_image = self.read_to_tensor(image_file)
        input_image, real_image = util.resize(input_image, real_image, 
                                                self.conf['LayoutGan']['ImgSize'])
        input_image, real_image = util.normalize(input_image, real_image)

        return input_image, real_image
