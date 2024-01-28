#include "ilqr/ilqr.h"

iLQR::iLQR(Dynamics* dynamic, Cost* cost, 
            double dt, int N, 
            float v_max, float omega_max)
{
    _dynamics = dynamic;
    _cost = cost;

    _dt = dt;
    _N = N;

    _v_max = v_max;
    _omega_max = omega_max;

    _x0 = VectorXd(_Nx);
    _xs = MatrixXd(_N+1, _Nx);
    _us = MatrixXd(_N, _Nu);

    _ks = MatrixXd::Zero(_N, _Nx);
    _Ks = Tensor<float, 3>(_N, _Nu, _Nx);

    _alphas = VectorXd(10);
    for (int i = 0; i < 10; ++i) 
    {
        _alphas(i) = std::pow(1.1, -std::pow(i, 2));
    }


    this->_rollout();
};

void iLQR::_rollout()
{
    int n, i;
    std::vector<float> xs(_xs.cols());
    std::vector<float> us(_us.cols());
    std::vector<DM> input, result;
    VectorXd xs_aux(3);


    _xs.row(0) = _x0;

    for (n=0; n<_N; n++)
    {
        for (i=0; i<_Nx; i++)
            xs.at(i) = _xs(n, i);

        for (i=0; i<_Nu; i++)
            us.at(i) = _us(n, i);

        input = {DM(xs), DM(us)};
        result = _dynamics->get_f(input);

        auto aux = static_cast<std::vector<float>>(result.at(0));
        _xs.row(n+1) << aux[0], aux[1], aux[2];

        ROS_INFO_STREAM("Xs: " << _xs.row(n+1) << "(n=" << n << ")");
    }
}