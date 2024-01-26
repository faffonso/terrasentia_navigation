#ifndef COST_H
#define COST_H

#include <iostream>
#include <string>

#include <ros/console.h>
#include <Eigen/Dense>
#include <casadi/casadi.hpp>

using namespace casadi;

typedef struct l_prime {
    double x;
} l_prime_t;

class Cost
{
    protected:
        int _N;
        SX _Qf, _Q, _R;
        Function _l, _lf, _l_x, _l_u, _l_xx, _l_uu, _l_prime;

    public:
        Cost(int N, 
            float Qf_x, float Qf_y, float Qf_theta,
            float Q_x, float Q_y, float Q_theta,
            float R_v, float R_omega);
        // ~Cost();

        std::vector<DM>  get_l_prime(std::vector<DM> input);
        Eigen::MatrixXf get_Qf();
};

#endif // DYNAMICS_H