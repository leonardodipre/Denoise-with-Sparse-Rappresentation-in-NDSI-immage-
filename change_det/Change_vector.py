import cv2
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt



class Vector_change():

    def __init__(self, image1, image2, inverse = False):
        self.image1 = image1
        self.image2 = image2
        self.inverse = inverse

    def perform_Vector_change(self, threshold):

        image1_traspose = self.image1.transpose(1, 2, 0)
        image2_traspose = self.image2.transpose(1, 2, 0)

    
        self.threshold = threshold
        
        

        c , n , m = self.image1.shape[0], self.image1.shape[1], self.image1.shape[2]
        
        
        image1_flat = image1_traspose.reshape( (n * m) , c)
        image2_flat = image2_traspose.reshape( (n * m), c)


        self.change_vector = np.abs(image1_flat - image2_flat)


         
        # Create a binary mask where changed pixels are 1 and unchanged pixels are 0
        binary_change_map = np.any(self.change_vector > self.threshold, axis=1).reshape(self.image1.shape[1], self.image1.shape[2])

        if self.inverse == False :
            self.output_image = np.ones_like(self.image1[0]) * 255  # Initialize with white (255) background

            self.output_image[binary_change_map] = 0  # Set changed pixels to black (0)
        else:
            self.output_image = np.ones_like(self.image1[0]) * 0  # Initialize with white (255) background

            self.output_image[binary_change_map] = 255  # Set changed pixels to black (0)


        return self.output_image