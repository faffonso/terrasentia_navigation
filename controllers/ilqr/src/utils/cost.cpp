#include "ilqr/cost.h"

Cost::Cost(int N, 
            float Qf_x, float Qf_y, float Qf_theta,
            float Q_x, float Q_y, float Q_theta,
            float R_v, float R_omega)
{
    _N = N;

    std::vector<double> Qf_values = {Qf_x, Qf_y, Qf_theta};
    std::vector<double> Q_values = {Q_x, Q_y, Q_theta};
    std::vector<double> R_values = {R_v, R_omega};

    _Qf = casadi::DM::diag(Qf_values);
    _Q = casadi::DM::diag(Q_values);
    _R = casadi::DM::diag(R_values);

    // State space
    SX x = SX::sym("x", 3);
    SX u = SX::sym("u", 2);

    SX l = 0.5 * (mtimes(x.T(), mtimes(_Q, x)) + mtimes(u.T(), mtimes(_R, u)));
    SX lf = 0.5 * (mtimes(x.T(), mtimes(_Qf, x)));

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
}

double Cost::trajectory_cost(std::vector<std::vector<double>> x, std::vector<std::vector<double>> u)
{
    std::size_t Nx = x.size();
    std::size_t Nu = u.size();

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


    double J = 0;
    std::vector<DM> input;

    for (int i=0; i<_N; i++)
    {
        input = {DM(x[i]), DM(u[i])};
        J += static_cast<double>(_l(input).at(0));
    }
    
    input = {DM(x[_N])};
    J += static_cast<double>(_lf(input).at(0));

    ROS_INFO_STREAM("Total Cost= "<< J);

    return J;
}

l_prime_t  Cost::get_l_prime(std::vector<DM> input)
{
    _l_prime.l_x = _l_x(input);
    _l_prime.l_u = _l_u(input);
    
    _l_prime.l_xx = _l_xx(input);
    _l_prime.l_uu = _l_uu(input);

    return _l_prime;
}

SX Cost::get_Qf()
{
    return _Qf;
}