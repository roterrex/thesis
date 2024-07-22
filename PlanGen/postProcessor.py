import skimage
from pixToPix import pix2Pix

class postProcessor:
    
    def __init__(self, conf, pix_model) -> None:
        self.conf = conf
        self.pix_model = pix_model


    def process(self, dataset):
        for image in dataset:
            image = skimage.color.rgb2hsv(image)
            image = self.flatten_color(image)
            self.show(image)
            image = self.denoise(image)
            self.show(image)
            data = self.ident_objects(image)



    def flatten_color(self, image):
        a=1
    def denoise(self, image):
        a=1
    def show(self, image, run_show):
        a=1
    def ident_objects(self, image):
        a=1

    

