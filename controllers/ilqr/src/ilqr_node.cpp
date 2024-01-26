#include <ros/ros.h>
#include <ros/console.h>
#include <iostream>

#include "ilqr/dynamics.h"
#include "ilqr/cost.h"

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

    Cost cost(5, 
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0);


    // input
    std::vector<double> x_v(3, 0);
    x_v[0] = 0.0;
    x_v[1] = 0.0;
    x_v[2] = 3.14/2;

    std::vector<double> u_v(2, 0);
    u_v[0] = 1.0;
    u_v[1] = 0.2;

    // Evaluate the function
    std::vector<casadi::DM> input = {DM(x_v), DM(u_v)};

    // // Print the result
    // std::cout << "Result of f: " << dynamic.get_f(input) << std::endl;
    // std::cout << "Result of fx: " << dynamic.get_f_x(input) << std::endl;
    // std::cout << "Result of fu: " << dynamic.get_f_u(input) << std::endl;

     
    while (ros::ok()) {
        ROS_INFO("Hello %s", "World");
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
