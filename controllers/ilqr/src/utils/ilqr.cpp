#include "ilqr/ilqr.h"

iLQR::iLQR(ros::NodeHandle nh, Dynamics* dynamic, Cost* cost, 
            double dt, int N, 
            float v_max, float omega_max)
{
    _nh = nh;
    _dynamic = dynamic;
    _cost = cost;

    _dt = dt;
    _N = N;

    _v_max = v_max;
    _omega_max = omega_max;

    _x    = VectorXd(_Nx);
    _x0   = VectorXd(_Nx);
    _xref = VectorXd(_Nx);

    _xs = MatrixXd(_N+1, _Nx);
    _us = MatrixXd(_N, _Nu);

    _xs_new = MatrixXd(_N+1, _Nx);
    _us_new = MatrixXd(_N, _Nu);

    _ks = MatrixXd::Zero(_N, _Nu);
    _Ks = Tensor<float, 3>(_N, _Nu, _Nx);

    _lambda = MatrixXd::Zero(_N+1, _p);

    _alphas = std::vector<float>(10);
    for (int i = 0; i < 10; ++i) 
        _alphas[i] = std::pow(1.1, -std::pow(i, 2));

    nh.getParam("/odom/frame_id", _odom_topic);
    nh.getParam("/odom/frame_id", _frame_id);

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
    _lambda = MatrixXd::Zero(_N+1, _p);

    double roll, pitch, yaw;

    tf::Quaternion q(_odom_msg.pose.pose.orientation.x,
                                    _odom_msg.pose.pose.orientation.y,
                                    _odom_msg.pose.pose.orientation.z,
                                    _odom_msg.pose.pose.orientation.w);

    tf::Matrix3x3 m(q);
    m.getRPY(roll, pitch, yaw);

    _x << _odom_msg.pose.pose.position.x,
            _odom_msg.pose.pose.position.y,
            yaw;

    tf::Quaternion q1(_goal_msg.pose.orientation.x,
                                    _goal_msg.pose.orientation.y,
                                    _goal_msg.pose.orientation.z,
                                    _goal_msg.pose.orientation.w);
    tf::Matrix3x3 m1(q1);
    m1.getRPY(roll, pitch, yaw);

    // _xref << _goal_msg.pose.position.x,
    //             _goal_msg.pose.position.y,
    //             yaw;

    _xref << 4.0, 4.0, 0.0;

    _x0 = _x - _xref;

    MatrixXd us(_N, _Nu);
    for (int n=0; n<_N; n++)
    {
        us(n, 0) = 0.0;
        us(n, 1) = 0.0;
    }

    //ROS_INFO_STREAM("Initial State" << _x0);
    //ROS_INFO_STREAM("Initial action control" << us);
    this->fit(us);

    _x0 = _x;
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

    //_cmd_vel_pub.publish(_cmd_vel_msg);

    ROS_INFO_STREAM("Run in " << ros::Time::now() - init);

}

void iLQR::fit(MatrixXd us)
{
    _us = us;

    this->_rollout();

    _J = _cost->trajectory_cost(_xs, _us, _lambda);
    //ROS_INFO_STREAM("Trajectory cost: " << _J);

    int max_iter = 100;
    float tol = 0.01;
    float regu = 20.0;
    float max_regu = 1000.0;
    float min_regu = 20.0;

    for (int iter=0; iter<max_iter; iter++)
    {
    
        regu = std::min(regu, max_regu);
        regu = std::max(regu, min_regu);

        this->_backward_pass(regu, _lambda); 

        if (std::abs(_delta_J) < tol)
        {
            //ROS_INFO_STREAM("Early stop");
            break;
        }

        for (float alpha : _alphas)
        {
            this->_forward_pass(alpha);
            _J_new = _cost->trajectory_cost(_xs_new, _us_new, _lambda);

            // ROS_INFO_STREAM("Trying converge " << _J_new << " on " << _J);

            if (std::abs(_J_new) < std::abs(_J))
            {
                _J = _J_new;
                _xs = _xs_new;
                _us = _us_new;

                regu *= 0.7;

                break;
            }
        }

        regu *= 2.0;
    }

    //ROS_INFO_STREAM("Final fit");
    //ROS_INFO_STREAM(_lambda);
    //ROS_INFO_STREAM(_us);
}

void iLQR::_forward_pass(float alpha)
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

void iLQR::_backward_pass(float regu, MatrixXd lambda)
{    
    int n, i, j;
    double J=0;

    f_prime_t f_prime;
    l_prime_t l_prime;
    c_prime_t c_prime;

    MatrixXd A, B, k, K;
    MatrixXd Q_x, Q_u, Q_xx, Q_uu, Q_ux; 
    MatrixXd Q_uu_reg, Q_ux_reg;
    MatrixXd c;

    std::vector<float> xs(_xs.cols());
    std::vector<float> us(_us.cols());
    std::vector<DM> input, result;

    MatrixXd regu_I = regu * MatrixXd::Identity(3, 3);
    MatrixXd I_mu = MatrixXd::Identity(_p, _p);

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

        Q_x.transposeInPlace();
        Q_u.transposeInPlace();

        Q_xx = l_prime.l_xx + f_prime.f_x.transpose() * S * f_prime.f_x;
        Q_uu = l_prime.l_uu + f_prime.f_u.transpose() * S * f_prime.f_u;
        Q_ux = f_prime.f_u.transpose() * S * f_prime.f_x;

        Q_uu_reg = Q_uu + f_prime.f_u.transpose() * regu_I * f_prime.f_u;
        Q_ux_reg = Q_ux + f_prime.f_u.transpose() * regu_I * f_prime.f_x;

        k = - Q_uu.inverse() * Q_u;
        K = - Q_uu.inverse() * Q_ux;

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

        //ROS_INFO_STREAM("Xs: " << _xs.row(n+1) << "(n=" << n << ")");
    }
}

void iLQR::_goal_callback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    _goal_msg = *msg;
}

void iLQR::_odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
    _odom_msg = *msg;
}