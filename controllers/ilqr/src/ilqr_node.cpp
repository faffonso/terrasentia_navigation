#include <ros/ros.h>
#include <ros/console.h>
#include <iostream>

#include "ilqr/dynamics.h"
#include "ilqr/cost.h"
#include "ilqr/ilqr.h"

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "ilqr_node");
    ros::NodeHandle nh;

    ros::Rate rate(1);

    float dt;
    std::string model;

    if(!nh.getParam("/controller/dynamics/dt", dt))
    {
        ROS_ERROR_STREAM("Sampling time (dt) could not be read.");
        return 0;
    }
    if(!nh.getParam("/controller/dynamics/model", model))
    {
        ROS_ERROR_STREAM("Dyamic model (model) could not be read.");
        return 0;
    }

    ROS_INFO_STREAM("Creating Dynamic Model using dt: " << dt << ", model: " << model);

    Dynamics dynamic(dt, model);


    int N;
    float Qf_x, Qf_y, Qf_theta, Q_x, Q_y, Q_theta, R_v, R_omega;

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

    ROS_INFO_STREAM("Creating Dynamic Model using N: " << N << ", final state cost: " << Q_x  << " " << Q_y <<  " "  << Q_theta << ", state cost: " << Q_x << " " << Q_y << " " << Q_theta << ", action control cost: " << R_v << " " << R_omega);

    Cost cost(N, 
            Qf_x, Qf_y, Qf_theta,
            Q_x, Q_y, Q_theta,
            R_v, R_omega);

    iLQR control(&dynamic, &cost,
                dt, N,
                1.0, 1.0);
    
    VectorXd x0(3);
    x0 << -4.0, -4.0, 0;

    MatrixXd us(N, 2);
    for (int n=0; n<N; n++)
        us(n, 0) = 1.0;

    control.fit(x0, us);

    while (ros::ok()) {
        ROS_INFO("Hello %s", "World");
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
