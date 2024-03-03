#include "ilqr/ilqr.h"

#define PI 3.1415

iLQR::iLQR(ros::NodeHandle nh, Dynamics* dynamic, Cost* cost, 
            double dt, int N)
{
    _nh = nh;
    _dynamic = dynamic;
    _cost = cost;

    _dt = dt;
    _N = N;

    _v_max = cost->get_v_max();
    _omega_max = cost->get_omega_max();

    _x    = VectorXd(_Nx);
    _x0   = VectorXd(_Nx);
    _xref = VectorXd(_Nx);

    _xs = MatrixXd(_N+1, _Nx);
    _us = MatrixXd(_N, _Nu);

    _xs_new = MatrixXd(_N+1, _Nx);
    _us_new = MatrixXd(_N, _Nu);

    _ks = MatrixXd::Zero(_N, _Nu);
    _Ks = Tensor<float, 3>(_N, _Nu, _Nx);

    _alphas = std::vector<float>(8);
    for (int i = 0; i < 8; ++i) 
        _alphas[i] = std::pow(0.5, i);

    nh.getParam("/odom/frame_id", _odom_topic);
    nh.getParam("/odom/frame_id", _frame_id);

    nh.getParam("/controller/iLQR/max_iter", _max_iter);
    nh.getParam("/controller/iLQR/tol", _tol);

    nh.getParam("/controller/iLQR/v_ref", _v_ref);

    _odom_msg.pose.pose.orientation.w = 1.0;

    _goal_msg.pose.position.x = 0.0;
    _goal_msg.pose.position.y = 0.0;
    _goal_msg.pose.orientation.w = 1.0;

    _goal_sub = _nh.subscribe("terrasentia/goal", 10, &iLQR::_goal_callback, this);
    _odom_sub = _nh.subscribe(_odom_topic, 10, &iLQR::_odom_callback, this);

    _path_pub = nh.advertise<nav_msgs::Path>("/terrasentia/path", 5);
    _cmd_vel_pub = nh.advertise<geometry_msgs::TwistStamped>("/terrasentia/cmd_vel", 5);  

}

void iLQR::run()
{
    ros::Time init = ros::Time::now();

    double odom_heading, goal_heading, heading;
    MatrixXd us(_N, _Nu);

    odom_heading = this->_get_heading(_odom_msg.pose.pose);
    goal_heading = this->_get_heading(_goal_msg.pose);

    _x0 << _odom_msg.pose.pose.position.x,
          _odom_msg.pose.pose.position.y,
          odom_heading;

    _xref << _goal_msg.pose.position.x,
             _goal_msg.pose.position.y,
             goal_heading;

    // us.setZero();

    heading = atan2(sin(odom_heading - goal_heading), cos(odom_heading - goal_heading));
    auto us0 = this->_single_shooting(heading);

    us.col(0).setConstant(_v_ref);
    us.col(1).setConstant(us0);


    this->fit(us);

    // Gen Path msg
    this->_rollout();

    geometry_msgs::PoseStamped pose;

    _path_msg.header.stamp = ros::Time::now();
    _path_msg.header.frame_id = _frame_id;


    for (int n=0; n<_N; n++)
    {
        geometry_msgs::Quaternion q_euler = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, _xs(n, 2));

        pose.pose.position.x = _xs(n, 0);
        pose.pose.position.y = _xs(n, 1);

        pose.pose.orientation = q_euler;

        _path_msg.poses.push_back(pose);
    }

    _path_pub.publish(_path_msg);
    _path_msg.poses.clear();

    _cmd_vel_msg.twist.linear.x = _us(0, 0);
    _cmd_vel_msg.twist.angular.z = _us(0, 1);

    _cmd_vel_pub.publish(_cmd_vel_msg);

    ROS_INFO_STREAM("Run iLQR | State " << _x0(0) << " " << _x0(1) << " " << _x0(2) << 
                            " | Reference " << _xref(0) << " " << _xref(1) << " " << _xref(2) << 
                            " | Action control " << _us(0, 0) << " " << _us(0, 1 ) << 
                            " | Time " << ros::Time::now() - init << " seconds");
}

void iLQR::fit(MatrixXd us)
{
    _us = us;

    this->_rollout();

    _J = _cost->trajectory_cost(_xs, _us, _xref);
    //ROS_INFO_STREAM("Trajectory cost: " << _J);

    for (int iter=0; iter<_max_iter; iter++)
    {

        this->_backward_pass(); 

        if (std::abs(_delta_J) < _tol)
        {
            //ROS_INFO_STREAM("Early stop");
            break;
        }

        for (float alpha : _alphas)
        {
            this->_forward_pass(alpha);
            _J_new = _cost->trajectory_cost(_xs_new, _us_new, _xref);

            //ROS_INFO_STREAM("Trying converge " << _J_new << " on " << _J << "Using regu " << regu);

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
    // ROS_INFO_STREAM(_xs);
    ROS_INFO_STREAM(_us);
}

void iLQR::_forward_pass(float alpha)
{
    int n, i, j;
    MatrixXd Ks(_Nu, _Nx), xs_aux(1, 3);

    std::vector<float> xs(_xs.cols());
    std::vector<float> us(_us.cols());
    std::vector<DM> input;

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
        auto result = _dynamic->get_f(input);

        _xs_new.row(n+1) << result(0, 0), result(0, 1), result(0, 2);

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

    MatrixXd k, K;
    MatrixXd Q_x, Q_u, Q_xx, Q_uu, Q_ux; 

    std::vector<float> xs(_xs.cols());
    std::vector<float> us(_us.cols());
    std::vector<DM> input;
    
    MatrixXd Qf = _cost->get_Qf();
    
    MatrixXd s = (Qf * _xs.row(_N).transpose() - _xref).transpose();
    MatrixXd S = Qf;

    for (n=_N-1; n>=0; n--)
    {
        for (i=0; i<_Nx; i++)
            xs.at(i) = _xs(n, i) - _xref(i);

        for (i=0; i<_Nu; i++)
            us.at(i) = _us(n, i);

        input = {DM(xs), DM(us)};
        f_prime = _dynamic->get_f_prime(input);
        l_prime = _cost->get_l_prime(input);

        Q_x = l_prime.l_x + s * f_prime.f_x;
        Q_u = l_prime.l_u + s * f_prime.f_u;

        Q_x.transposeInPlace();
        Q_u.transposeInPlace();

        Q_xx = l_prime.l_xx + f_prime.f_x.transpose() * S * f_prime.f_x;
        Q_uu = l_prime.l_uu + f_prime.f_u.transpose() * S * f_prime.f_u;
        Q_ux = l_prime.l_ux + f_prime.f_u.transpose() * S * f_prime.f_x;

        k = - Q_uu.completeOrthogonalDecomposition().pseudoInverse() * Q_u;
        K = - Q_uu.completeOrthogonalDecomposition().pseudoInverse() * Q_ux;

        _ks.row(n) = k.transpose();
        for (i=0; i<_Nu; i++)
            for (j=0; j<_Nx; j++)
                _Ks(n, i, j) = K(i, j);

        s = Q_x + K.transpose() * Q_u + Q_ux.transpose() * k + K.transpose() * Q_uu * k;
        S = Q_xx + K.transpose() * Q_uu * K + K.transpose() * Q_ux + Q_ux.transpose() * K;

        s.transposeInPlace();

        J += (0.5 * (k.transpose() * Q_uu * k) + k.transpose() * Q_u)(0,0);
    }

    _delta_J = J;
    //ROS_INFO_STREAM("Final backward cost: " << J);

}

void iLQR::_rollout()
{
    int n, i;
    std::vector<float> xs(_xs.cols());
    std::vector<float> us(_us.cols());
    std::vector<DM> input;


    _xs.row(0) = _x0;

    for (n=0; n<_N; n++)
    {
        for (i=0; i<_Nx; i++)
            xs.at(i) = _xs(n, i);

        for (i=0; i<_Nu; i++)
            us.at(i) = _us(n, i);

        input = {DM(xs), DM(us)};
        auto result = _dynamic->get_f(input);
        
        _xs.row(n+1) << result(0, 0), result(0, 1), result(0, 2);
        //ROS_INFO_STREAM("Xs: " << _xs.row(n+1) << "(n=" << n << ")");
    }
}

double iLQR::_single_shooting(double err_heading)
{  
    double abs_err = abs(err_heading);
    double res = ( exp(_shoot * abs_err) - 1) * (_cost->get_omega_max() / exp(_shoot * PI));

    return (err_heading >= 0) ? -res : res;
}

double iLQR::_get_heading(geometry_msgs::Pose pose)
{    
    double roll, pitch, yaw;

    tf::Quaternion q(pose.orientation.x,
                     pose.orientation.y,
                     pose.orientation.z,
                     pose.orientation.w);

    tf::Matrix3x3 m(q);
    m.getRPY(roll, pitch, yaw);

    return yaw;
}

void iLQR::_goal_callback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    _goal_msg = *msg;
}

void iLQR::_odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
    _odom_msg = *msg;
}