#!/usr/bin/env python3

from __future__ import print_function

import rospy

from inference.srv import RTInferenceService, RTInferenceServiceResponse

def callback(req):
    print(req)

def main():
    rospy.init_node("test_node")

    service = rospy.Service("RTInference", RTInferenceService, callback)

    rospy.spin()
    
if __name__ == '__main__':
    main()