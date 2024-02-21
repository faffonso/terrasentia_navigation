#ifndef COST_H
#define COST_H

/**
 * @file cost.h
 * @brief Implementation of quadratic cost function interface
 *
 * This file contains the implementation of a quadratic cost function,
 * including running cost Jacobians and Hessians, as well as a function
 * to calculate trajectory cost.
 *
 * Authors: Francisco Affonso Pinto
 *          
 */

/* Bibs */
#include <iostream>
#include <string>

#include <ros/console.h>
#include <Eigen/Dense>
#include <casadi/casadi.hpp>

using namespace casadi;
using namespace Eigen;

/* Struct */

typedef struct l_prime {
    MatrixXd l_x;   // dl/dx
    MatrixXd l_u;   // dl/du
    MatrixXd l_xx;  // d^2 l / dx^2
    MatrixXd l_uu;  // d^2 l / du^2 
} l_prime_t;

class Cost
{
    protected:
        int _N,  // Precition horizon size
            _Nx, // State vector size
            _Nu; // Action control vector size

        l_prime_t _l_prime; // Struct contains cost funcion derivatives
        SX _Qf, _Q, _R;     // Weight matrices
        
        Function _l, _lf, _l_x, _l_u, _l_xx, _l_uu; // CasADi functions of cost interface

    public:
        /**
         * @brief Construct a new Cost object
         * 
         */
        Cost();

        /**
         * @brief Construct a new Cost object
         * 
         * @param N Prediction Horizon
         * @param Qf_x Final cost of x
         * @param Qf_y Final cost of y
         * @param Qf_theta Final cost of theta
         * @param Q_x Cost of x
         * @param Q_y Cost of y
         * @param Q_theta Cost of theta
         * @param R_v Cost of linear velocity
         * @param R_omega Cost of angular velocity
         */
        Cost(int N, 
            float Qf_x, float Qf_y, float Qf_theta,
            float Q_x, float Q_y, float Q_theta,
            float R_v, float R_omega);

        /**
         * @brief Get the l prime object
         * 
         * @param input State and Action control {x, u}
         * @return l_prime_t 
         */
        l_prime_t  get_l_prime(std::vector<DM> input);

        /**
         * @brief Get trajectory cost
         * 
         * @param x State 
         * @param u Action control
         * @return double 
         */
        double trajectory_cost(MatrixXd x, MatrixXd u);

        /**
         * @brief Get the Qf object
         * 
         * @return MatrixXd 
         */
        MatrixXd get_Qf();
};

#endif // COST_H