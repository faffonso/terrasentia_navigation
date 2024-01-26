#ifndef DYNAMICS_H
#define DYNAMICS_H

#include <iostream> 
#include <string>

#include <ros/console.h>
#include <casadi/casadi.hpp>

using namespace casadi;

class Dynamics
{
    protected:
        float _dt;
        std::string _model;
        Function _f, _f_x, _f_u;

    public:
        Dynamics(double dt, std::string model);
        // ~Dynamics();
        std::vector<casadi::DM>  get_f(std::vector<DM> input);
        std::vector<casadi::DM>  get_f_x(std::vector<DM> input);
        std::vector<casadi::DM>  get_f_u(std::vector<DM> input);
};

#endif // DYNAMICS_H