#include "ilqr/dynamics.h"

Dynamics::Dynamics(double dt, std::string model)
{
    // Saving classes atributes
    _dt = dt;
    _model = model;

    // State space
    SX x = SX::sym("x", 3);
    SX u = SX::sym("u", 2);

    SX f;

    // Unicycle Kinematic Model
    if (model == "standard") 
    {
        SX x1_dot = u(0) * cos(x(2));
        SX x2_dot = u(0) * sin(x(2));
        SX x3_dot = u(1);
        
        SX x_dot = vertcat(x1_dot, x2_dot, x3_dot);

        f = x + x_dot * dt;
    }

    // Error-Tracking (Mobile point)
    else if (model == "error-tracking") 
    {
        SX e1_dot =  u(1) * x(1) - u(0);
        SX e2_dot = -u(1) * x(0);
        SX e3_dot = -u(1);

        SX e_dot = vertcat(e1_dot, e2_dot, e3_dot);

        f = x + e_dot * dt;
    }
    else {
        ROS_ERROR_STREAM("Not exist this dynamic model" << model);
    }

    // Dynamics derivatives (Aproxximation for A and B --> x_dot = Ax + Bu)
    SX f_x = jacobian(f, x);
    SX f_u = jacobian(f, u);

    // Dynamics Functions
    _f = Function("f", {x, u}, {f});
    _f_x = Function("f_x", {x, u}, {f_x});
    _f_u = Function("f_u", {x, u}, {f_u});
}

std::vector<DM> Dynamics::get_f(std::vector<DM> input) {
    return _f(input);
}

std::vector<DM> Dynamics::get_f_x(std::vector<DM> input) {
    return _f_x(input);
}

std::vector<DM> Dynamics::get_f_u(std::vector<DM> input) {
    return _f_u(input);
}
