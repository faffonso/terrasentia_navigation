#include <ros/ros.h>
#include <ros/console.h>
#include <iostream>

#include "ilqr/dynamics.h"
#include "ilqr/cost.h"
#include "ilqr/ilqr.h"

bool initDynamics(Dynamics& dynamic, ros::NodeHandle& nh);
bool initCost(Cost& cost, ros::NodeHandle& nh);

// ANSI color codes
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"

/*
Include params to Cost and Dynamic object
Debug Optimization after rotation
Test max values
*/

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "ilqr_node");
    ros::NodeHandle nh;

    ros::Rate rate(20);

    Dynamics dynamic;
    if (!initDynamics(dynamic, nh)) {
        return 1;
    }

    Cost cost;
    if (!initCost(cost, nh)) {
        return 1; 
    }

    int N;
    float dt;

    nh.getParam("/controller/dynamics/dt", dt);
    nh.getParam("/controller/cost/N", N);

    iLQR control(nh, &dynamic, &cost,
                dt, N);
    
    ROS_INFO_STREAM(GREEN << "iLQR Created!" << RESET);

    while (ros::ok()) {
        control.run();
        rate.sleep();
        ros::spinOnce();
    }

    return 0;
}

bool initDynamics(Dynamics& dynamic, ros::NodeHandle& nh) {
    float dt;
    std::string model;

    if (!nh.getParam("/controller/dynamics/dt", dt)) {
        ROS_ERROR_STREAM("Sampling time (dt) could not be read.");
        return false;
    }
    if (!nh.getParam("/controller/dynamics/model", model)) {
        ROS_ERROR_STREAM("Dynamic model (model) could not be read.");
        return false;
    }

    ROS_INFO_STREAM(GREEN << "Dynamic Model Object Created!" << RESET);

    dynamic = Dynamics(dt, model);
    return true;
}

bool initCost(Cost& cost, ros::NodeHandle& nh) {
    int N, eps, t;
    float Qf_x, Qf_y, Qf_theta, Q_x, Q_y, Q_theta, R_v, R_omega, v_max, omega_max;

    if(!nh.getParam("/controller/cost/N", N))
    {
        ROS_ERROR_STREAM("Prediction Horizon (N) could not be read.");
        return 0;
    }
    if(!nh.getParam("/controller/cost/Qf_x", Qf_x))
    {
        ROS_ERROR_STREAM("Final state x cost (Qf_x) could not be read.");
        return 0;
    }
    if(!nh.getParam("/controller/cost/Qf_y", Qf_y))
    {
        ROS_ERROR_STREAM("Final state y cost (Qf_y) could not be read.");
        return 0;
    }
    if(!nh.getParam("/controller/cost/Qf_theta", Qf_theta))
    {
        ROS_ERROR_STREAM("Final state theta cost (Qf_theta) could not be read.");
        return 0;
    }
    if(!nh.getParam("/controller/cost/Q_x", Q_x))
    {
        ROS_ERROR_STREAM("State x cost (Q_x) could not be read.");
        return 0;
    }
    if(!nh.getParam("/controller/cost/Q_y", Q_y))
    {
        ROS_ERROR_STREAM("State y cost (Q_y) could not be read.");
        return 0;
    }
    if(!nh.getParam("/controller/cost/Q_theta", Q_theta))
    {
        ROS_ERROR_STREAM("State theta cost (Q_theta) could not be read.");
        return 0;
    }
    if(!nh.getParam("/controller/cost/R_v", R_v))
    {
        ROS_ERROR_STREAM("Linear velocity cost (R_v) could not be read.");
        return 0;
    }
    if(!nh.getParam("/controller/cost/R_omega", R_omega))
    {
        ROS_ERROR_STREAM("Angular velocity cost (R_omega) could not be read.");
        return 0;
    }

    if(!nh.getParam("/controller/cost/v_max", v_max))
    {
        ROS_ERROR_STREAM("Max linear speed (v_max) could not be read.");
        return 0;
    }

        if(!nh.getParam("/controller/cost/omega_max", omega_max))
    {
        ROS_ERROR_STREAM("Max angular spped (omega_max) could not be read.");
        return 0;
    }

        if(!nh.getParam("/controller/cost/eps", eps))
    {
        ROS_ERROR_STREAM("Barrier Function eps could not be read.");
        return 0;
    }

        if(!nh.getParam("/controller/cost/t", t))
    {
        ROS_ERROR_STREAM("Barrier Function t could not be read.");
        return 0;
    }

    ROS_INFO_STREAM(GREEN << "Cost Object Created!" << RESET);

    cost = Cost(N, Qf_x, Qf_y, Qf_theta, Q_x, Q_y, Q_theta, R_v, R_omega, v_max, omega_max, eps, t);
    return true;
}
