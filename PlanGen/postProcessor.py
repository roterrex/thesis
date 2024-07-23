import skimage
import cv2
from pixToPix import pix2Pix
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

class postProcessor:    
    hue_inc_list = [0,1,2,4,6,8,11]
    hue_inc_list_0_1 = np.divide([0,1,2,4,6,8,11],12)
    hue_exc_list = [3,5,7,9,10]
    def __init__(self, conf, pix_model) -> None:
        self.conf = conf
        self.pix_model = pix_model


    def process(self, dataset):
        steps = dataset.cardinality().numpy()
        if not os.path.isdir(self.conf['PostProcess']['OOI_save_path']): 
            os.makedirs(self.conf['PostProcess']['OOI_save_path']) 
        for step, (input_image, target) in dataset.take(steps).enumerate():
            image = self.get_image(input_image)[0]*0.5+0.5 # get a predicted image
            imageHSV = skimage.color.rgb2hsv(image) # convert to hsv
            imageFlat = self.flatten_color(imageHSV) # correct to a limited set of colors
            hueLayers = self.threshhold(imageFlat) #split the image into layers by hue
            layersNoNoise = self.denoise(hueLayers) # remove holes and too small areas per hue
            data = self.ident_objects(layersNoNoise) # convert to csv
            if self.conf['PostProcess']['plot_images']:
                imageNoNoise = skimage.color.hsv2rgb(self.de_threshhold(layersNoNoise)) #split the image into layers by hue (plot only)
                self.show(target[0]*0.5+0.5, image, skimage.color.hsv2rgb(imageFlat), imageNoNoise)


            data.to_csv(f"{self.conf['PostProcess']['OOI_save_path']}/{step}.csv")

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
                    y[0] = (round(y[0]*12) % 12)
                    if y[0] in self.hue_exc_list:
                        y[0] = 0
                        y[1] = 0
                    else:
                        y[0] /= 12
                        y[1] = 1
                    y[2] = 1
        return img
    def denoise(self, layers):
        lays = np.empty_like(layers)
        np.copyto(lays, layers)
        for i in range(lays.shape[0]):
            lays[i] = skimage.morphology.area_opening(lays[i], area_threshold=self.conf['PostProcess']['close_thresh'])
        return lays
    
    def show(self, imageIn, imageHSV, imageFlat, imageNoNoise):
        display_list = [imageIn, imageHSV, imageFlat, imageNoNoise]
        title = ['Input Image', 'generated', 'flattened', 'denoised']

        plt.figure(figsize=(15, 8))
        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i])
            plt.axis('off')
        plt.pause(0.1)
            
    def threshhold(self, image):
        imgs = np.empty_like(image, shape=[len(self.hue_inc_list),image.shape[0],image.shape[1]])
        
        for i in range(len(self.hue_inc_list)):
            img = np.empty_like(image)
            np.copyto(img, image)
            imgs[i] = np.logical_and(img[:,:,0] == self.hue_inc_list_0_1[i], img[:,:,1] == 1)
        return imgs

    def de_threshhold(self, image):
        img = np.ones_like(image, shape=[image.shape[1],image.shape[2], 3])
        img[:,:,0] = 0

        for i in range(len(self.hue_inc_list)):
            img[:,:,0] += np.multiply(image[i], self.hue_inc_list_0_1[i])
        img[:,:,1] = np.logical_or(img[:,:,0] != 0, image[0] != 0) #set white if hue not 0(red) and pixel is not red
        return img

    def ident_objects(self, image):
        cols=['Type', 'box_left', 'box_top', 'Width', 'Height', 'Centroid_x', 'Centroid_y', 'Volume']
        oois = pd.DataFrame(columns=cols)
        for i in range(len(self.hue_inc_list)):
            (totalLabels, label_ids, val, cent) = \
                    cv2.connectedComponentsWithStats((image[i]*255).astype(np.uint8), 4, cv2.CV_8U)
            exclude = val[:,4].argmax()
            totalLabels -= 1
            val = np.delete(val, exclude, 0)
            cent = np.delete(cent, exclude, 0)

            data = [np.ones(totalLabels)*self.hue_inc_list[i], 
                    val[:,cv2.CC_STAT_TOP], val[:,cv2.CC_STAT_LEFT], 
                    val[:,cv2.CC_STAT_WIDTH], val[:,cv2.CC_STAT_HEIGHT], 
                    cent[:,0], cent[:,1], 
                    val[:,cv2.CC_STAT_AREA]
                    ]
            oois = pd.concat([oois, pd.DataFrame(np.transpose(data), columns=oois.columns)], ignore_index=True)

        return oois

    

