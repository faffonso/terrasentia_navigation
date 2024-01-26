#include <ros/ros.h>
#include <ros/console.h>
#include <iostream>

#include <string>
#include <casadi/casadi.hpp>

using namespace casadi;

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "ilqr_node");
    ros::NodeHandle nh;

    ros::Rate rate(1);

    SX x = SX::sym("x", 3);
    std::cout << "State: " << x << std::endl;

    SX u = SX::sym("u", 2);
    std::cout << "Action control: " << u << std::endl;
    

    SX x1_dot = u(0) + x(2);
    SX x2_dot = u(0) + x(2);
    SX x3_dot = u(1);

    SX x_dot = vertcat(x1_dot, x2_dot, x3_dot);

    std::cout << "State Space " << x_dot << std::endl;

    Function f = Function("f", {x, u}, {x_dot});

    // input
    std::vector<double> x_v(3, 0);
    x_v[0] = 10.0;
    x_v[1] = 5.0;
    x_v[2] = 5.0;

    std::vector<double> u_v(2, 0);
    u_v[0] = 5.0;
    u_v[1] = 3.0;

    std::cout << "Result of x: " << x_v << std::endl;
    std::cout << "Result of u: " << u_v << std::endl;

    // Evaluate the function
    std::vector<casadi::DM> input = {DM(x_v), DM(u_v)};
    std::vector<casadi::DM> result = f(input);

    // Print the result
    std::cout << "Result of f: " << result << std::endl;

     
    while (ros::ok()) {
        ROS_INFO("Hello %s", "World");
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
