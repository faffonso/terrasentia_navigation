#!/usr/bin/env python3

from __future__ import print_function

from wp_gen.srv import WpGen, WpGenResponse, WpGenRequest
from wp_gen.msg import CropLine

import rospy

def callback(req):
    left_line = CropLine(-2.25, 380.00)
    right_line = CropLine(-2.25, 566.86)

    return WpGenResponse(left_line, right_line)

def add_two_ints_server():
    rospy.init_node('WpGen')
    s = rospy.Service('WpGen', WpGen, callback)
    rospy.spin()

if __name__ == "__main__":
    add_two_ints_server()