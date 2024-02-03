#!/usr/bin/env python

import rospy
from your_package_name.srv import RTInferenceService

def client():
    rospy.init_node('client_node')
    rospy.wait_for_service('rt_inference_service')
    
    try:
        # Create a service proxy
        rt_inference_service = rospy.ServiceProxy('rt_inference_service', RTInferenceService)
        
        # Create a request object
        request = RTInferenceServiceRequest()
        
        # Call the service with the request
        response = rt_inference_service(request)
        
        # Access the response fields
        m1 = response.m1
        m2 = response.m2
        b1 = response.b1
        b2 = response.b2
        
        rospy.loginfo(f"Client received a response: m1={m1}, m2={m2}, b1={b1}, b2={b2}")
        
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)

if __name__ == "__main__":
    client()