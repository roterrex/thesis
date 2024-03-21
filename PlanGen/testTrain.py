import tensorflow as tf
import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display


class testTrain:
    log_dir="logs/"
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    def __init__(self, pix_model, dataloader, conf) -> None:
        self.conf = conf
        self.pix_model = pix_model
        self.dataloader = dataloader
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.pix_model.generator_optimizer,
                                 discriminator_optimizer=self.pix_model.discriminator_optimizer,
                                 generator=self.pix_model.generator,
                                 discriminator=self.pix_model.discriminator)






    def generate_images(self, test_input, tar, plot=False):
        prediction = self.pix_model.generator(test_input, training=True)

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        if plot:
            plt.figure(figsize=(15, 15))
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(title[i])
                # Getting the pixel values in the [0, 1] range to plot.
                plt.imshow(display_list[i] * 0.5 + 0.5)
                plt.axis('off')
            plt.pause(0.1)


    @tf.function
    def train_step(self, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.pix_model.generator(input_image, training=True)

            disc_real_output = self.pix_model.discriminator([input_image, target], training=True)
            disc_generated_output = self.pix_model.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.pix_model.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.pix_model.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.pix_model.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.pix_model.discriminator.trainable_variables)

        self.pix_model.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.pix_model.generator.trainable_variables))
        self.pix_model.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.pix_model.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

    def fit(self, train_ds, test_ds, steps):
        example_input, example_target = next(iter(test_ds.take(1)))
        start = time.time()
        steps += 1
        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if (step) % 1000 == 0:
                display.clear_output(wait=True)

            if (step) % 1000 == 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')
                start = time.time()
                print(f"Step: {step//1000}k")


            self.generate_images(example_input, example_target, (step) % 1000 == 0)


            self.train_step(input_image, target, step)

            # Training step
            if (step+1) % 50 == 0:
                print('.', end='', flush=True)


            # Save (checkpoint) the model every 5k steps
            if (step + 1) % 5000 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            if step == steps-1:
                break