#!/usr/bin/env python3

import rospy
import numpy as np

from module.cost import Cost
from module.dynamics import Dynamics
from module.ilqr import iLQR

def main():
    rospy.init_node('ilqr_controller')

    rate = rospy.Rate(100) 

    dynamic_p = rospy.get_param("controller/dynamics")
    dynamic = Dynamics(**dynamic_p)

    cost_p = rospy.get_param("controller/cost")
    cost = Cost(**cost_p)

    control_p = rospy.get_param("controller/iLQR")
    control = iLQR(dynamic, cost, **control_p)
    
    while not rospy.is_shutdown():
        control.run()
        rate.sleep()

if __name__ == '__main__':
    main()