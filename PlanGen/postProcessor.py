import skimage
from pixToPix import pix2Pix
from matplotlib import pyplot as plt
import numpy as np

class postProcessor:    
    def __init__(self, conf, pix_model) -> None:
        self.conf = conf
        self.pix_model = pix_model


    def process(self, dataset):
        steps = dataset.cardinality().numpy()
        steps
        for step, (input_image, target) in dataset.take(steps).enumerate():
            image = self.get_image(input_image)[0] # get a predicted image
            imageHSV = skimage.color.rgb2hsv(image*0.5+0.5) # convert to hsv
            imageFlat = self.flatten_color(imageHSV) # correct to a limited set of colors
            imageNoNoise = self.denoise(imageFlat) # remove holes and too small areas
            self.show(imageHSV, imageFlat, skimage.color.rgb2hsv(input_image[0]*0.5+0.5), True)
            data = self.ident_objects(imageNoNoise) # convert to csv

            if step == steps-1:
                break

    def get_image(self, ds_input):
        return self.pix_model.generator(ds_input, training=True)

    def flatten_color(self, image):
        img = np.empty_like(image)
        np.copyto(img, image)
        for x in image:
            for y in x:
                y[1] *0.5
        for x in img:
            for y in x:
                if y[1] < 0.2 :#and y[2] > 0.90:
                    y[0] = 0
                    y[1] = 0
                    y[2] = 1
                else:
                    y[0] = (round(y[0]*12) % 12) / 12
                    y[1] = 1
                    y[2] = 1
        return img
    def denoise(self, image):
        img = np.empty_like(image)
        np.copyto(img, image)
        return img
    def show(self, imageHSV, imageFlat, imageNoNoise, plot):
        if plot:
            display_list = [imageHSV, imageFlat, imageNoNoise]
            title = ['Input Image', 'flattened', 'denoised']

            plt.figure(figsize=(15, 8))
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(title[i])
                # Getting the pixel values in the [0, 1] range to plot.
                plt.imshow(skimage.color.hsv2rgb(display_list[i]))
                plt.axis('off')
            plt.pause(0.1)
            
    def ident_objects(self, image):
        a=1

    

