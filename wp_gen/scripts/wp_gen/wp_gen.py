#!/usr/bin/env python3

import rospy

import numpy as np
import matplotlib.pyplot as plt
import cv2

class Wp_gen():
    def __init__(self, row_height, row_width, img_height, img_width):
        
        self.row_height = row_height
        self.row_width = row_width

        self.img_height = img_height
        self.img_width = img_width

    def convert_origin(self, m1, m2, c1, c2):

        widht_min = -self.img_width/2
        widht_max = self.img_width/2

        x = np.linspace(widht_min, widht_max, 100) 

        y1 = -m1 * x - (c1 + self.img_width)
        y2 = -m2 * x - (c2 + self.img_width)
        
        return x, y1, y2

