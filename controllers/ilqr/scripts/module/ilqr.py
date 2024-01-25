#!/usr/bin/env python3

import rospy
import time

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistStamped, PoseStamped

from tf.transformations import euler_from_quaternion, quaternion_from_euler

from module.cost import *
from module.dynamics import *

np.set_printoptions(suppress=True)


class iLQR():
    def __init__(self, dynamics, cost, dt=0.1, N=10, v_max=1.0, omega_max=1.0, regu_init=20, regu_max=1000):
        """Constructs an iLQR controller.

        Args:
            dynamics: Dynamics System.
            cost: Cost function.
            dt: Sampling time
            N: Horizon length.
        """

        self.dynamics = dynamics
        self.cost = cost

        self.N = N
        self.alpha = 1.0
        self.regu_init = regu_init
        self.regu_max = regu_max

        self.v_max = v_max
        self.omega_max = omega_max

        self._k = np.zeros((N, 3))
        self._K = np.zeros((N, 2, 3))

        # Publishers and Subscribers
        self.odom_topic = rospy.get_param("odom/frame_id")
        self.odom_subscriber = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        self.goal_subscriber = rospy.Subscriber("/terrasentia/goal", PoseStamped, self.goal_callback)

        self.path_publisher     = rospy.Publisher("/terrasentia/path", Path, queue_size=1)
        self.cmd_vel_publisher  = rospy.Publisher("/terrasentia/cmd_vel", TwistStamped, queue_size=10)

        self.odom    = Odometry()
        self.goal    = PoseStamped()
        self.cmd_vel = TwistStamped()

    def run(self):
        start_time = time.time()

        x_0 = self.state_from_pose(self.odom.pose.pose)
        xref = self.state_from_pose(self.goal.pose)

        x0 = x_0 - xref

        us = np.zeros((self.N, 2))

        x, u = self.fit(x0, us)

        x += xref

        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.odom_topic
        for xi in x:
            pose = self.pose_from_state(xi)
            path_msg.poses.append(pose)

        self.cmd_vel.twist.linear.x = u[0][0]
        self.cmd_vel.twist.angular.z = u[0][1]

        self.path_publisher.publish(path_msg)
        self.cmd_vel_publisher.publish(self.cmd_vel)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'iLQR Run | State {x0} | Reference {xref} | Action control {u[0]} | Time {elapsed_time:.3f} seconds')

    def fit(self, x0, us, n_iter=10, tol=1e-6):
        """Compute optimal control

        Args:
            x0: Initial state.
            u0: Initial control path.
            n_iter: Max iterations.
            tol: Tolerance. 

        Returns:
            xs: Optimal state path.
            us: Optimal control path.
        """
        alphas = 1.1**(-np.arange(10)**2)

        regu = self.regu_init

        # Rollout for first J
        xs, J_old = self._rollout(x0, us)

        # Action control compute loop
        for iteration in range(n_iter):
            k, K, _ = self._backward_pass(xs, us, regu)

            for alpha in alphas:
                xs_new, us_new = self._forward_pass(xs, us, k, K, alpha)
                J = self.cost._trajectory_cost(xs_new, us_new)

                # print(f'Trying convergence {J} on {J_old}')
                
                if (abs(J) < abs(J_old)):

                    J_old = J
                    xs = xs_new
                    us = us_new
                    regu *= 0.7

                    break
            
            # Increase regularization
            else:
                regu *= 2.0

        return xs, us

    def _forward_pass(self, xs, us, ks, Ks, alpha, hessian=False):
        """Apply the forward dynamics

        Args:
            xs: Initial state.
            us: Control path.
            ks: Feedforward gains.
            Ks: Feedback gains.
            alpha: Line search coefficient.
            
        Returns:
            Tuple of:
                x_new: New state path.
                u_new: New action control.
        """
        
        u_new = np.empty((self.N, 2))
        x_new = np.empty((self.N+1, 3))

        x_new[0] = np.copy(xs[0])

        # Compute new action and state control
        for i in range (self.N):

            #print(alpha * ks[i])
            u_new[i] = us[i] + Ks[i] @ (x_new[i] - xs[i]) + alpha * ks[i]

            if (u_new[i][0] < 0):
                u_new[i][0] = 0
            elif (u_new[i][0] > self.v_max):
                u_new[i][0] = self.v_max

            if (u_new[i][1] < -self.omega_max):
                u_new[i][1] = -self.omega_max
            elif (u_new[i][1] > self.omega_max):
                u_new[i][1] = self.omega_max

            x_new[i+1] = np.array(self.dynamics.f(x_new[i], u_new[i])).T

        # print(u_new)
        # print(x_new)

        return x_new, u_new



    def _backward_pass(self, xs, us, regu):
        """Computes the feedforward and feedback gains k and K.

        Args:
            xs: Initial state.
            us: Control path.

        Returns:
            ks: feedforward gains.
            Ks: feedback gains.
            J: Cost path.
        """
        
        N = self.N

        ks = np.empty(us.shape)
        Ks = np.empty((us.shape[0], us.shape[1], xs.shape[1]))

        J = 0

        Qf = self.cost.get_Qf()

        s = Qf @ xs[N-1]
        S = Qf

        regu_I = regu * np.eye(2)

        for n in range(N - 1, -1, -1):

            # Obtain f and l derivatives
            Ak, Bk = self.dynamics.f_prime(xs[n], us[n])
            l_x, l_u, l_xx, l_ux, l_uu  = self.cost.l_prime(xs[n], us[n])

            # Q_terms
            Q_x  = l_x  + np.dot(s, Ak)
            Q_u  = l_u  + np.dot(s, Bk)
            Q_xx = l_xx + np.dot(Ak.T, np.dot(S, Ak))
            Q_uu = l_uu + np.dot(Bk.T, np.dot(S, Bk))
            Q_ux = l_ux + np.dot(Bk.T, np.dot(S, Ak))

            # Regularization
            Q_uu_reg = Q_uu + regu_I       

            # Feedforward and feedback gains
            k = np.dot(-np.linalg.inv(Q_uu), Q_u)
            K = np.dot(-np.linalg.inv(Q_uu), Q_ux)

            ks[n], Ks[n] = k, K

            # V_terms
            s = Q_x + K.T @ Q_u + Q_ux.T @ k + K.T @ Q_uu @ k
            S = Q_xx + K.T@Q_ux + K.T @ Q_ux + K.T @ Q_uu @ K

            #Sum cost 
            J += 1/2 * np.dot(k.T, np.dot(Q_uu, k)) + np.dot(k, Q_u)

        return ks, Ks, J
        
    def _rollout(self, x0, us):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            x0: Initial state.
            us: Control path.

        Returns:
            J: Cost path.
        """
        J = 0
        N = self.N

        xs = np.empty((N+1, 3))
        xs[0] = x0

        # Calculate path state over a x0 and us
        for n in range(N):
            xs[n+1] = np.array(self.dynamics.f(xs[n], us[n])).T

            J += self.cost._l(xs[n], us[n])[0]
            # print(f'Index {n} | cost: {J}')

        J += self.cost._lf(xs[N-1])
        # print(f'Index {N-1} | cost: {J}')

        return xs, J

    def state_from_pose(self, pose):
        position      = pose.position
        orientation_q = pose.orientation
        orientation = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]

        x_0 = position.x
        y_0 = position.y
        _, _, theta_0 = euler_from_quaternion(orientation)

        return np.array([x_0, y_0, theta_0])

    def pose_from_state(self, x):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.odom_topic
   
        pose_msg.pose.position.x = x[0]
        pose_msg.pose.position.y = x[1] 

        orientation = quaternion_from_euler(0.0, 0.0, x[2])
        pose_msg.pose.orientation.x = orientation[0]
        pose_msg.pose.orientation.y = orientation[1]
        pose_msg.pose.orientation.z = orientation[2]
        pose_msg.pose.orientation.w = orientation[3]

        return pose_msg
    
    def odom_callback(self, msg):
        self.odom = msg

    def goal_callback(self, msg):
        print(msg)
        self.goal = msg
