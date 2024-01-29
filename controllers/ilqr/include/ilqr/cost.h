#ifndef COST_H
#define COST_H

#include <iostream>
#include <string>

#include <ros/console.h>
#include <Eigen/Dense>
#include <casadi/casadi.hpp>

using namespace casadi;
using namespace Eigen;

typedef struct l_prime {
    MatrixXd l_x;
    MatrixXd l_u;
    MatrixXd l_xx;
    MatrixXd l_uu;
} l_prime_t;

class Cost
{
    protected:
        int _N, _Nx, _Nu;
        l_prime_t _l_prime;
        SX _Qf, _Q, _R;
        Function _l, _lf, _l_x, _l_u, _l_xx, _l_uu;

    public:
        Cost();
        Cost(int N, 
            float Qf_x, float Qf_y, float Qf_theta,
            float Q_x, float Q_y, float Q_theta,
            float R_v, float R_omega);

        l_prime_t  get_l_prime(std::vector<DM> input);
        double trajectory_cost(MatrixXd x, MatrixXd u);
        MatrixXd get_Qf();
};

#endif // COST_H