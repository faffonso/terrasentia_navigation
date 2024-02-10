#include "ilqr/cost.h"

Cost::Cost()
{
    _N = 10;
    _Nx = 3;
    _Nu = 2;
}

Cost::Cost(int N, 
            float Qf_x, float Qf_y, float Qf_theta,
            float Q_x, float Q_y, float Q_theta,
            float R_v, float R_omega,
            float v_max, float omega_max)
{
    _N = N;

    _Nx = 3;
    _Nu = 2;

    _I_mu = MatrixXd::Identity(_p, _p);
    _lambda = std::vector<float>(_p, 0);

    std::vector<double> Qf_values = {Qf_x, Qf_y, Qf_theta};
    std::vector<double> Q_values = {Q_x, Q_y, Q_theta};
    std::vector<double> R_values = {R_v, R_omega};

    _Qf = casadi::DM::diag(Qf_values);
    _Q = casadi::DM::diag(Q_values);
    _R = casadi::DM::diag(R_values);

    SX x = SX::sym("x", 3);
    SX u = SX::sym("u", 2);

    SX l = (mtimes(x.T(), mtimes(_Q, x)) + mtimes(u.T(), mtimes(_R, u)));
    SX lf = (mtimes(x.T(), mtimes(_Qf, x)));

    SX l_x = jacobian(l, x);
    SX l_u = jacobian(l, u);

    SX l_xx = jacobian(l_x, x);
    SX l_uu = jacobian(l_u, u);

    _l = Function("l", {x, u}, {l});
    _lf = Function("lf", {x}, {lf});
    _l_x = Function("l_x", {x, u}, {l_x});
    _l_u = Function("l_u", {x, u}, {l_u});
    _l_xx = Function("l_xx", {x, u}, {l_xx});
    _l_uu = Function("l_uu", {x, u}, {l_uu});

    // Constraints Functions
    SX c1 = v_max - u(0);
    SX c2 = omega_max - u(1);

    SX c = vertcat(c1, c2);

    SX c_x = jacobian(c, x);
    SX c_u = jacobian(c, u);

    _c = Function("c", {x, u}, {c});
    _c_x = Function("c_x", {x, u}, {c_x});
    _c_u = Function("c_u", {x, u}, {c_u});
}

double Cost::trajectory_cost(MatrixXd x, MatrixXd u, std::vector<float>& lambda)
{
    std::size_t Nx = x.rows();
    std::size_t Nu = u.rows();

    if (Nu != _N)
    {
        ROS_ERROR_STREAM("Prediction Size (N) is diferente of action control size (Nu) - (" << _N << "!=" << Nu << ").");
        return -1;
    }

    if (Nu != (Nx -1))
    {
        ROS_ERROR_STREAM("State size (Nx) or Action size (Nu) have wrong size - (Nx=" << Nx << "and Nu=" << Nu << ").");
        return -1;
    }


    int n, i;
    double J = 0;
    std::vector<float> xs(x.cols());
    std::vector<float> us(u.cols());
    std::vector<DM> input, result;

    for (n=0; n<_N; n++)
    {
        for (i=0; i<_Nx; i++)
            xs.at(i) = x(n, i);

        for (i=0; i<_Nu; i++)
            us.at(i) = u(n, i);

        input = {DM(xs), DM(us)};
        J += static_cast<double>(_l(input).at(0));

        auto Jc = _c(input).at(0);

        for (i=0; i<_p; i++)
        {
            auto j = static_cast<double>(Jc(i));
            //J += 0.5 * (j * _I_mu(i, i) * j + lambda[i] * j);

            lambda[i] = std::max(0.0, lambda[i] + j);
        }
    }
    
    for (i=0; i<_Nx; i++)
            xs.at(i) = (float)x(_N, i);

    input = {DM(xs)};
    //J += static_cast<double>(_lf(input).at(0));

    return J;
}

l_prime_t  Cost::get_l_prime(std::vector<DM> input)
{
    _l_prime.l_x = Eigen::Matrix<double, 1, 3>::Map(DM::densify(_l_x(input).at(0)).nonzeros().data(), 1, 3);
    _l_prime.l_u = Eigen::Matrix<double, 1, 2>::Map(DM::densify(_l_u(input).at(0)).nonzeros().data(), 1, 2);
    
    _l_prime.l_xx = Eigen::Matrix<double, 3, 3>::Map(DM::densify(_l_xx(input).at(0)).nonzeros().data(), 3, 3);
    _l_prime.l_uu = Eigen::Matrix<double, 2, 2>::Map(DM::densify(_l_uu(input).at(0)).nonzeros().data(), 2, 2);

    return _l_prime;
}

MatrixXd Cost::get_c(std::vector<DM> input) {
    return Eigen::Matrix<double, 2, 1>::Map(DM::densify(_c(input).at(0)).nonzeros().data(), 2, 1);
}

c_prime_t Cost::get_c_prime(std::vector<DM> input){
    _c_prime.c_x = Eigen::Matrix<double, 3, 2>::Map(DM::densify(_c_x(input).at(0)).nonzeros().data(), 3, 2);
    _c_prime.c_u = Eigen::Matrix<double, 2, 2>::Map(DM::densify(_c_u(input).at(0)).nonzeros().data(), 2, 2);

    return _c_prime;
}

MatrixXd Cost::get_Qf()
{
    MatrixXd Qf(3, 3);    
    Qf = Eigen::Matrix<double, 3, 3>::Map(DM::densify(_Qf).nonzeros().data(), 3, 3);

    return Qf;
}