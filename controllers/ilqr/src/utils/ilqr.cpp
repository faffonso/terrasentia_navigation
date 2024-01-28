#include "ilqr/ilqr.h"

iLQR::iLQR(Dynamics* dynamic, Cost* cost, 
            double dt, int N, 
            float v_max, float omega_max)
{
    _dynamic = dynamic;
    _cost = cost;

    _dt = dt;
    _N = N;

    _v_max = v_max;
    _omega_max = omega_max;

    _x0 = VectorXd(_Nx);

    _xs = MatrixXd(_N+1, _Nx);
    _us = MatrixXd(_N, _Nu);

    _xs_new = MatrixXd(_N+1, _Nx);
    _us_new = MatrixXd(_N, _Nu);

    _ks = MatrixXd::Zero(_N, _Nu);
    _Ks = Tensor<float, 3>(_N, _Nu, _Nx);

    _alphas = std::vector<float>(10);
    for (int i = 0; i < 10; ++i) 
        _alphas[i] = std::pow(1.1, -std::pow(i, 2));

}

void iLQR::fit(VectorXd x0, MatrixXd us)
{
    _x0 = x0;
    _us = us;

    this->_rollout();
    //_J = _cost->trajectory_cost(_xs, _us);

    int max_iter = 10;

    for (int iter=0; iter<max_iter; iter++)
    {
        ROS_INFO_STREAM("Iteration: " << iter);
        this->_backward_pass();

        for (float alpha : _alphas)
        {
            ROS_INFO_STREAM("Alpha: " << alpha);
            this->_forward_pass(alpha);
            //_J_new = _cost->trajectory_cost(_xs_new, _us_new);

            if (std::abs(_J_new) < std::abs(_J))
            {
                _J = _J_new;
                _xs = _xs_new;
                _us = _us_new;

                break;
            }
        }
    }

    ROS_INFO_STREAM("Final fit");
    ROS_INFO_STREAM(_xs);
    ROS_INFO_STREAM(_us);
}

void iLQR::_forward_pass(double alpha)
{
    int n, i, j;
    MatrixXd Ks(_Nu, _Nx), xs_aux(1, 3);

    std::vector<float> xs(_xs.cols());
    std::vector<float> us(_us.cols());
    std::vector<DM> input, result;

    _xs_new.row(0) = _xs.row(0);

    for (n=0; n<_N; n++)
    {
        for (i=0; i<_Nu; i++)
            for (j=0; j<_Nx; j++)
                Ks(i, j) = _Ks(n, i, j);

        xs_aux = (_xs_new.row(n) - _xs.row(n)).transpose();

        _us_new.row(n) = _us.row(n) + (Ks * xs_aux).transpose() + alpha * _ks.row(n);

        for (i=0; i<_Nx; i++)
            xs.at(i) = _xs_new(n, i);

        for (i=0; i<_Nu; i++)
            us.at(i) = _us_new(n, i);

        input = {DM(xs), DM(us)};
        result = _dynamic->get_f(input);

        auto aux = static_cast<std::vector<float>>(result.at(0));
        _xs_new.row(n+1) << aux[0], aux[1], aux[2];

        //ROS_INFO_STREAM("Xs: " << _xs_new.row(n+1) << "(n=" << n << ")");
        //ROS_INFO_STREAM("Us: " << _us_new.row(n) << "(n=" << n << ")");
    }
}

void iLQR::_backward_pass()
{
    int n, i, j;
    double J=0;

    f_prime_t f_prime;
    l_prime_t l_prime;

    MatrixXd A, B, k, K;
    MatrixXd Q_x, Q_u, Q_xx, Q_uu, Q_ux; 

    std::vector<float> xs(_xs.cols());
    std::vector<float> us(_us.cols());
    std::vector<DM> input, result;
    
    MatrixXd Qf = _cost->get_Qf();
    
    MatrixXd s = (Qf * _xs.row(_N).transpose()).transpose();
    MatrixXd S = Qf;

    for (n=_N-1; n>=0; n--)
    {
        for (i=0; i<_Nx; i++)
            xs.at(i) = _xs(n, i);

        for (i=0; i<_Nu; i++)
            us.at(i) = _us(n, i);

        input = {DM(xs), DM(us)};
        f_prime = _dynamic->get_f_prime(input);
        l_prime = _cost->get_l_prime(input);

        Q_x = l_prime.l_x + s * f_prime.f_x;
        Q_u = l_prime.l_u + s * f_prime.f_u;

        Q_x = Q_x.transpose();
        Q_u = Q_u.transpose();

        Q_xx = l_prime.l_xx + f_prime.f_x.transpose() * S * f_prime.f_x;
        Q_uu = l_prime.l_uu + f_prime.f_u.transpose() * S * f_prime.f_u;
        Q_ux = f_prime.f_u.transpose() * S * f_prime.f_x;

        k = - Q_uu.inverse() * Q_u;
        K = - Q_uu.inverse() * Q_ux;


        _ks.row(n) = k.transpose();
        for (i=0; i<_Nu; i++)
            for (j=0; j<_Nx; j++)
                _Ks(n, i, j) = K(i, j);

        s = Q_x + K.transpose() * Q_u + Q_ux.transpose() * k + K.transpose() * Q_uu * k;
        S = Q_xx + K.transpose() * Q_uu * K + K.transpose() * Q_ux + Q_ux.transpose() * K;

        s = s.transpose();

        J += (0.5 * (k.transpose() * Q_uu * k) + k.transpose() * Q_u)(0,0);
    }

    _delta_J = J;
    ROS_INFO_STREAM("Final backward cost: " << J);

}

void iLQR::_rollout()
{
    int n, i;
    std::vector<float> xs(_xs.cols());
    std::vector<float> us(_us.cols());
    std::vector<DM> input, result;


    _xs.row(0) = _x0;

    for (n=0; n<_N; n++)
    {
        for (i=0; i<_Nx; i++)
            xs.at(i) = _xs(n, i);

        for (i=0; i<_Nu; i++)
            us.at(i) = _us(n, i);

        input = {DM(xs), DM(us)};
        result = _dynamic->get_f(input);

        auto aux = static_cast<std::vector<float>>(result.at(0));
        _xs.row(n+1) << aux[0], aux[1], aux[2];

        ROS_INFO_STREAM("Xs: " << _xs.row(n+1) << "(n=" << n << ")");
    }
}