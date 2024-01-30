#!/usr/bin/env python3

import rospy

import numpy as np
import matplotlib.pyplot as plt
import cv2

from wp_gen.wp_gen import *

def main():
    rospy.init_node('wp_gen_node')

    rate = rospy.Rate(100) 

    m1 = 4.33
    m2 = 4.33
    b1 = -108.38
    b2 = -419.56

    wp_gen = Wp_gen(7.5, 1.35, 244, 244)

    x, y1, y2 = wp_gen.convert_origin(m1, m2, b1, b2)
 
    image = cv2.imread('../figures/test2.jpeg')

    plt.figure()
    height, width, _ = image.shape
    extent = [-width/2, width/2, 0, height]
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), extent=extent)

    #Plot the lines
    plt.plot(x, y1, label='$y_1 = m_1\cdot x + b_1$')  # Line 1
    plt.plot(x, y2, label='$y_2 = m_2 \cdot x + b_2$')  # Line 2


    plt.xlim(-width/2, width/2)  
    plt.ylim(0, height) 

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of Lines')
    plt.legend()    

    # Show the plot
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()

    
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    main()