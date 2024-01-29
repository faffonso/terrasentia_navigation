#ifndef ILQR_H
#define ILQR_H

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
        float _dt, _v_max, _omega_max;
        int _N, _Nx=3, _Nu=2;
        double _J=0, _J_new=0, _delta_J=0;
        std::string _frame_id, _odom_topic;

        ros::NodeHandle _nh;
        ros::Subscriber _odom_sub, _goal_sub;
        ros::Publisher _cmd_vel_pub, _path_pub; 


        Dynamics* _dynamic;
        Cost* _cost;

        geometry_msgs::PoseStamped _goal_msg;
        nav_msgs::Odometry _odom_msg;

        nav_msgs::Path _path_msg;
        geometry_msgs::TwistStamped _cmd_vel_msg;

        std::vector<float> _alphas;

        VectorXd _x0, _xref, _x;
        MatrixXd _u0, _ks, _xs, _us, _xs_new, _us_new;
        Tensor<float, 3> _Ks;


        void _forward_pass(float alpha);
        void _backward_pass(float regu);
        void _rollout(); 

        void _goal_callback(const geometry_msgs::PoseStamped::ConstPtr& msg);
        void _odom_callback(const nav_msgs::Odometry::ConstPtr& msg);

    public:
        iLQR(ros::NodeHandle nh, Dynamics* dynamic, Cost* cost, 
            double dt, int N, 
            float v_max, float omega_max);

        void fit(MatrixXd us);
        void run();
        
};

#endif // ILQR_H

