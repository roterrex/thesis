import tensorflow as tf

class util:
    @staticmethod
    def resize(input_image, real_image, size):
        input_image = tf.image.resize(input_image, [size[0], size[1]],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [size[0], size[1]],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image

    @staticmethod
    def random_crop(input_image, real_image, size):
        #size = conf['LayoutGan']['ImgSize']
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, size[0], size[1], size[2]])

        return cropped_image[0], cropped_image[1]

    # Normalizing the images to [-1, 1]
    @staticmethod
    def normalize(input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image
    
    @staticmethod
    @tf.function()
    def random_jitter(input_image, real_image, conf):
        size = conf['LayoutGan']['ImgSize']
        # Resizing to 286x286
        input_image, real_image = util.resize(input_image, real_image, [int(size[0]*1.2), int(size[1]*1.2)])

        # Random cropping back to 256x256
        input_image, real_image = util.random_crop(input_image, real_image, size)

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image