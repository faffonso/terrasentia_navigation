#include "ilqr/dynamics.h"

Dynamics::Dynamics()
{
    _dt = 0.1;
    _model = "";
}

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

f_prime_t Dynamics::get_f_prime(std::vector<DM> input)
{
    _f_prime.f_x = Eigen::Matrix<double, 3, 3>::Map(DM::densify(_f_x(input).at(0)).nonzeros().data(), 3, 3);
    _f_prime.f_u = Eigen::Matrix<double, 3, 2>::Map(DM::densify(_f_u(input).at(0)).nonzeros().data(), 3, 2);

    return _f_prime;
}
