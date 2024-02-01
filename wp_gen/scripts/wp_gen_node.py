#!/usr/bin/env python3

import rospy

import numpy as np
import matplotlib.pyplot as plt
import cv2

from wp_gen.wp_gen import *

def main():
    rospy.init_node('wp_gen_node')

    rate = rospy.Rate(10) 
    wp_gen = Wp_gen(7.5, 1.35, 2.5, 244, 244)

    while not rospy.is_shutdown():
        wp_gen.run()
        rate.sleep()

if __name__ == '__main__':
    main()