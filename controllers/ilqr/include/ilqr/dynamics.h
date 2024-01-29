#ifndef DYNAMICS_H
#define DYNAMICS_H

#include <iostream> 
#include <string>

#include <ros/console.h>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>


using namespace casadi;
using namespace Eigen;

typedef struct f_prime {
    MatrixXd f_x;
    MatrixXd f_u;
} f_prime_t;
class Dynamics
{
    protected:
        float _dt;
        std::string _model;
        f_prime_t _f_prime;
        Function _f, _f_x, _f_u;

    public:
        Dynamics();
        Dynamics(double dt, std::string model);
        std::vector<casadi::DM>  get_f(std::vector<DM> input);
        f_prime_t  get_f_prime(std::vector<DM> input);
};

#endif // DYNAMICS_H