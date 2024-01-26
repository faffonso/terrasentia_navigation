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

    ROS_INFO_STREAM("Qf: " << _Qf);
    ROS_INFO_STREAM("Q: " << _Q);
    ROS_INFO_STREAM("R: " << _R);

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
    _lf = Function("lf", {x, u}, {lf});
    _l_x = Function("l_x", {x, u}, {l_x});
    _l_u = Function("l_u", {x, u}, {l_u});
    _l_xx = Function("l_xx", {x, u}, {l_xx});
    _l_uu = Function("l_uu", {x, u}, {l_uu});

    // input
    std::vector<double> x_v(3, 0);
    x_v[0] = 0.0;
    x_v[1] = 0.0;
    x_v[2] = 3.14/2;

    std::vector<double> u_v(2, 0);
    u_v[0] = 1.0;
    u_v[1] = 0.2;

    // Evaluate the function
    std::vector<casadi::DM> input = {DM(x_v), DM(u_v)};

    ROS_INFO_STREAM("t1: " << _l(input));
    ROS_INFO_STREAM("t2: " << _lf(input));
    ROS_INFO_STREAM("t3: " << _l_x(input));
    ROS_INFO_STREAM("t4: " << _l_u(input));
    ROS_INFO_STREAM("t1: " << _l_xx(input));
    ROS_INFO_STREAM("t1: " << _l_uu(input));


}