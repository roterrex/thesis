import skimage
from pixToPix import pix2Pix

class postProcessor:    
    def __init__(self, conf, pix_model) -> None:
        self.conf = conf
        self.pix_model = pix_model


    def process(self, dataset):
        for step, (input_image, target) in dataset.enumerate():
            image = self.get_image(input_image) # get a predicted image
            image = skimage.color.rgb2hsv(image) # convert to hsv
            image = self.flatten_color(image) # correct to a limited set of colors
            self.show(image)
            image = self.denoise(image) # remove holes and too small areas
            self.show(image)
            data = self.ident_objects(image) # convert to csv

    def get_image(self, ds_input):
        return self.pix_model.generator(ds_input, training=True)

    def flatten_color(self, image):
        for x in image:
            for y in image:
                image[x][y][hue] = (round(image[x][y][hue]/30) % 12) * 30
    def denoise(self, image):
        a=1
    def show(self, image, run_show):
        a=1
    def ident_objects(self, image):
        a=1

    

