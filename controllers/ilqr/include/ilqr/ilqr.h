#ifndef ILQR_H
#define ILQR_H

#include <iostream> 
#include <string>
#include <vector>

#include <ros/console.h>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "ilqr/dynamics.h"
#include "ilqr/cost.h"

using namespace casadi;
using namespace Eigen;

class iLQR
{
    protected:
        float _dt, _v_max, _omega_max;
        int _N, _Nx=3, _Nu=2;
        double J=0, J_new=0;

        Dynamics* _dynamics;
        Cost* _cost;

        VectorXd _x0, _alphas;
        MatrixXd _u0, _ks, _xs, _us, _xs_new, _us_new;
        Tensor<float, 3> _Ks;


        void _forward_pass(double alpha);
        /*
        void _forward_pass(std::vector<DM> xs,
                            std::vector<DM> us,
                            std::vector<double> ks,
                            std::vector<double> Ks,
                            std::vector<double> alpha);

        void _backward_pass(std::vector<DM> xs,
                            std::vector<DM> us,
                            float regu);
*/
        void _rollout(); 


    public:
        iLQR(Dynamics* dynamic, Cost* cost, 
            double dt, int N, 
            float v_max, float omega_max);

        // void run();
        // std::vector<DM> fit(std::vector<DM> x0, 
        //                         std::vector<DM> u0,
        //                         int max_iter, double tol);

        
};

#endif // ILQR_H

