from matplotlib import pyplot as plt
import tensorflow as tf
from keras import metrics
import numpy as np
import os

class pix2Pix:
    def __init__(self, conf):
        self.conf = conf

        self.shape = self.conf['LayoutGan']['ImgSize']
        self.output_channels = 3
        self.LAMBDA = 100

        
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.Generator()
        self.Discriminator()

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.loss_method_Cross = self.conf['Loss']['Loss_function'] == 'cross'
        self.loss_method_Wesser = self.conf['Loss']['Loss_function'] == 'wesser'
        if not (self.loss_method_Cross or self.loss_method_Wesser):
            print("p2p : no loss method selected")

        self.checkpoint_dir = self.conf['Checkpoint']['save_dir']
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
                generator_optimizer=self.generator_optimizer,
                discriminator_optimizer=self.discriminator_optimizer,
                generator=self.generator,
                discriminator=self.discriminator
            )
        


    
    def Generator(self):
        inputs = tf.keras.layers.Input(shape=self.shape)
        
        down_stack = [
            self.downsample(256, 4, apply_batchnorm=False),     # (batch_size, (out) 32, 32, 256)
            self.downsample(512, 4),                            # (batch_size, (out) 16, 16, 512)
            self.downsample(512, 4),                            # (batch_size, (out) 8, 8, 512)
            self.downsample(512, 4),                            # (batch_size, (out) 4, 4, 512)
            self.downsample(512, 4),                            # (batch_size, (out) 2, 2, 512)
            self.downsample(512, 4),                            # (batch_size, (out) 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True),          # (batch_size, (out) 2, 2, 1024)
            self.upsample(512, 4, apply_dropout=True),          # (batch_size, (out) 4, 4, 1024)
            self.upsample(512, 4, apply_dropout=True),          # (batch_size, (out) 8, 8, 1024)
            self.upsample(512, 4),                              # (batch_size, (out) 16, 16, 1024)
            self.upsample(256, 4),                              # (batch_size, (out) 32, 32, 512)
            self.upsample(128, 4),                              # (batch_size, (out) 64, 64, 256)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.output_channels, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='tanh')  # (batch_size, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        self.generator = tf.keras.Model(inputs=inputs, outputs=x)
    
    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=self.shape, name='input_image')
        tar = tf.keras.layers.Input(shape=self.shape, name='target_image')

        inp_tar = tf.keras.layers.concatenate([inp, tar])                           # (batch_size, 256, 256, channels*2)

        down = self.downsample(256, 4, apply_batchnorm=False)(inp_tar)              # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down)                          # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1)                  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)                     # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer, activation='linear')(zero_pad2)    # (batch_size, 30, 30, 1)

        self.discriminator = tf.keras.Model(inputs=[inp, tar], outputs=last)

    def generator_loss(self, disc_generated_output, gen_output, targets):
        #gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        #l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        if self.loss_method_Cross:
            total_gen_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)#gan_loss + (self.LAMBDA * l1_loss)
        elif self.loss_method_Wesser:
            total_gen_loss = -tf.reduce_mean(disc_generated_output)

        return total_gen_loss#, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        if self.loss_method_Cross:
            real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
            generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        elif self.loss_method_Wesser:
            real_loss = tf.reduce_mean(disc_real_output)
            generated_loss = tf.reduce_mean(disc_generated_output)

        total_disc_loss = generated_loss - real_loss

        return total_disc_loss
    
    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result
    
    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result
    
    def load_checkpoint(self, run_load):
        if run_load:
            if self.conf['Checkpoint']['checkpoint_to_load'] == '':
                check_point_path = tf.train.latest_checkpoint(self.conf['Checkpoint']['load_dir'])
            else:
                check_point_path = self.conf['Checkpoint']['load_dir']+'\\'+self.conf['Checkpoint']['checkpoint_to_load']
            print("load from checkpoint : ", check_point_path)
            self.checkpoint.restore(check_point_path)

    def save_checkpoint(self, run_save):
        if run_save:
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            print("Checkpoint Saved")



    """
    # implementation of wasserstein loss
    def wasserstein_loss(self, y_true, y_pred):
        print(y_pred)
        print(y_true * y_pred)
        m = tf.keras.metrics.Mean()
        m.update_state(y_true * y_pred)
        return m.result()
    """
