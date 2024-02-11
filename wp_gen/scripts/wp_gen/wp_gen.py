#!/usr/bin/env python3

from __future__ import print_function

from wp_gen.srv import RTInference, RTInferenceResponse, RTInferenceRequest

import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import numpy as np
import colorful as cf
import math

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

class Wp_gen():
    def __init__(self, row_height, row_width, D, img_height, img_width):
        """ 
        Waypoint generator

        Generates waypoints using reference corn lines predicted using
        a neural network applied to point clouds (LiDAR).
        The lines are in the following format: y = mx + b

        Attributes:
            row_height: Real height of the image in meters
            row_width: Real width of the image in meters 
            img_height: Image height resolution in pixels
            img_width: Image width resolution in pixels
            odom_sub: ROS subscriber for odometry messages
            goal_pub: ROS publisher for goal messages
            D: Euclidean distance of robot to waypoints
        """

        self.row_height = row_height
        self.row_width = row_width

        self.img_height = img_height
        self.img_width = img_width

        odom_topic = rospy.get_param("wp_gen/odom/topic")
        frame_id = rospy.get_param("wp_gen/odom/frame_id")


        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        self.goal_pub = rospy.Publisher('/terrasentia/goal', PoseStamped, queue_size=10)

        self.odom_msg = Odometry()
        self.goal_msg = PoseStamped()
        self.goal_msg.header.frame_id = frame_id
        
        self.D = D

        rospy.loginfo(cf.green(f"Client created!"))


    def run(self, show=False, verbose=False):
        rospy.wait_for_service('RTInference')
        rospy.loginfo(cf.orange(f"Send request to server!"))

        try:
            service_proxy = rospy.ServiceProxy('RTInference', RTInference)
            response = service_proxy(show)
            rospy.loginfo(cf.yellow(f"Response received!"))
            if (verbose == True):
                rospy.loginfo(response)

        except rospy.ServiceException as e:
            print("Service call failed:", e)
        
        m1 = response.left_line.m
        c1 = response.left_line.b

        m2 = response.right_line.m
        c2 = response.right_line.b

        rospy.loginfo(cf.orange(f'Line1 m={m1}, b={c1}'))
        rospy.loginfo(cf.orange(f'Line2 m={m2}, b={c2}'))

        x, y = self.get_target(m1, m2, c1, c2)

        x *= self.row_width / self.img_width
        y *= self.row_height / self.img_height

        rospy.loginfo(cf.orange(f'Y={y}, X={x}'))

        q = (
            self.odom_msg.pose.pose.orientation.x,
            self.odom_msg.pose.pose.orientation.y,
            self.odom_msg.pose.pose.orientation.z,
            self.odom_msg.pose.pose.orientation.w)
        _, _, heading_global = euler_from_quaternion(q)

        heading = np.arctan2(x, y)

        self.goal_msg.pose.position.x = self.odom_msg.pose.pose.position.x + x * np.sin(heading_global) + y * np.cos(heading_global)
        self.goal_msg.pose.position.y = self.odom_msg.pose.pose.position.y - (x * np.cos(heading_global) - y * np.sin(heading_global))

        q = quaternion_from_euler(0.0, 0.0, - heading + heading_global)

        self.goal_msg.pose.orientation.x = q[0]
        self.goal_msg.pose.orientation.y = q[1]
        self.goal_msg.pose.orientation.z = q[2]
        self.goal_msg.pose.orientation.w = q[3]

        self.goal_pub.publish(self.goal_msg)    

        if verbose == True:
            print(f'New waypoint x={x} y={y}, heading={heading + heading_global} -- heading{heading} + global_heading{heading_global}')    

    def get_target(self, m1, m2, c1, c2):
        """ 
        Get Target

        Generates a point with robot pose as reference.

        Args:
            m1: Angular coefficient of the first corn line
            m2: Angular coefficient of the second corn line
            c1: Linear coefficient of the first corn line
            c2: Linear coefficient of the second corn line
        Returns:
            x: x-coordinate of the generated point
            y: y-coordinate of the generated point
        """

        # Convert coefficient args to a reference line to follow up
        m, c = self._convert_origin(m1, m2, c1, c2)

        # Get x and y resolution [m/px]^2
        x_regu = (self.row_width / self.img_width)**2
        y_regu = (self.row_height / self.img_height)**2
        
        # Solve equation using Euclidean distance and line equation
        x1, x2 = self._solve_quadratic(
                    m**2 + (x_regu/y_regu),
                    2 * m * c,
                    c**2 - self.D**2/y_regu)
        
        # Get corresponding y and return the possible value
        y1 = x1 * m + c
        y2 = x2 * m + c

        if (y1 >= 0 and y1 <= self.img_height):
            return x1, y1
        elif (y2 >= 0 and y2 <= self.img_height):
            return x2, y2
        else:
            rospy.logerr("Waypoint doesn't match with crop lines")
            return None


    def odom_callback(self, msg):
        self.odom_msg = msg

    def _convert_origin(self, m1, m2, c1, c2):

        widht_min = -self.img_width/2
        widht_max = self.img_width/2

        m = -(m1 + m2) / 2
        c = -(c1 + c2) / 2 
        c -= self.img_width

        return m, c

    
    def _solve_quadratic(self, a, b, c):
        discriminant = (b**2) - (4*a*c)

        sol1 = (-b - math.sqrt(discriminant)) / (2 * a)
        sol2 = (-b + math.sqrt(discriminant)) / (2 * a)

        return sol1, sol2

