#ifndef ILQR_H
#define ILQR_H

/**
 * @file ilqr.h
 * @brief Implementation of iLQR controller
 *
 * This file contains the implementation of iLQR controller.
 *
 * Authors: Francisco Affonso Pinto
 *          
 */

/* Bibs */
#include <iostream> 
#include <string>
#include <vector>

#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_datatypes.h>

#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Path.h"
#include "geometry_msgs/TwistStamped.h"
#include "nav_msgs/Odometry.h"

#include "ilqr/dynamics.h"
#include "ilqr/cost.h"

using namespace casadi;
using namespace Eigen;

class iLQR
{
    protected:
        int _N,     // Precition horizon size
            _Nx=3,  // State vector size
            _Nu=2;  // Action control vector size
        
        float _dt,      // Sampling time [s]
            _v_max,     // Max linear velocity [m/s]
            _omega_max; // Max angular velocity [rad/s]

        double _J=0,    // Actual trajectory cost
            _J_new=0,   // New trajectory cost
            _delta_J=0; // Backward pass variation cost

        std::string _frame_id,  // Global frame id
                _odom_topic;    // Odometry topic



        // Child objects
        Dynamics* _dynamic;
        Cost* _cost;

        // ROS handlers, publishers and subscribers
        ros::NodeHandle _nh;                   
        ros::Subscriber _odom_sub, _goal_sub;  
        ros::Publisher _cmd_vel_pub, _path_pub;

        // ROS msgs
        geometry_msgs::PoseStamped _goal_msg;
        nav_msgs::Odometry _odom_msg;
        nav_msgs::Path _path_msg;
        geometry_msgs::TwistStamped _cmd_vel_msg;

        VectorXd _x0,   // Initial state
                _xref,  // Refernece state
                _x;     // Input state (x0 - xref)

        // Auxiliar Matrices and Vectors to comunicate between methods
        MatrixXd _u0, _ks, _xs, _us, _xs_new, _us_new;
        Tensor<float, 3> _Ks;
        std::vector<float> _alphas;

        /**
         * @brief Simulate the system using (x_0, us_new) to get the trajectory xs_new
         * 
         * Calculate the new action control (us_new) using k and K calculated in
         * backward pass and use it to simualte the new trajectory (xs_new)
         * 
         * @param alpha Gain for k
         */
        void _forward_pass(float alpha);

        /**
         * @brief Calculate k and K gains estimating the trajectory cost
         * 
         * @param regu Regularization term
         */
        void _backward_pass(float regu);

        /**
         * @brief Simulate the system using (x_0, us) to get the trajectory xs
         * 
         */
        void _rollout(); 

        // ROS Subscriber callbacks
        void _goal_callback(const geometry_msgs::PoseStamped::ConstPtr& msg);
        void _odom_callback(const nav_msgs::Odometry::ConstPtr& msg);

    public:
        /**
         * @brief Construct a new i L Q R object
         * 
         * @param nh Node Handler
         * @param dynamic System dynamics
         * @param cost Cost function interface
         * @param dt Sampling time [s]
         * @param N Prediction Horizon
         * @param v_max Max linear velocity
         * @param omega_max Max angular velocity
         */
        iLQR(ros::NodeHandle nh, Dynamics* dynamic, Cost* cost, 
            double dt, int N, 
            float v_max, float omega_max);

        /**
         * @brief Run controller
         * 
         */
        void run();

        /**
         * @brief Execute iLQR steps to find a optimal control
         * 
         * @param us Optimal action control
         */
        void fit(MatrixXd us);
        
};

#endif // ILQR_H
