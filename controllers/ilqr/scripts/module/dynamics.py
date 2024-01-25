#!/usr/bin/env python3

import rospy

import numpy as np
import casadi as ca

PI = 3.14

class Dynamics():
    def __init__(self, dt=.1, model="standard"):

        self.model = model
        self.dt = dt

        # State vector
        x = ca.SX.sym('x', 3)
            ## Standard
            # x_1 = x
            # x_2 = y
            # x_3 = theta 
            ## Tracking-error
            # e_1 = longitudinal error
            # e_2 = lateral error
            # e_3 = rotational error
        
        # Action control vector
        u = ca.SX.sym('u', 2) 
            # u_1 = linear velocity
            # u_2 = angular velocity

        # Dynamic function
        if (model == "standard"):

            x1_dot = u[0] * np.cos(x[2])
            x2_dot = u[0] * np.sin(x[2])
            x3_dot = u[1]
            
            x_dot = ca.veccat(x1_dot, x2_dot, x3_dot)

            f = x + x_dot * dt

        ## Tracking-error
        elif (model == "error-tracking"):

            e1_dot =  u[1] * x[1] - u[0]
            e2_dot = -u[1] * x[0]
            e3_dot = -u[1]

            e_dot = ca.veccat(e1_dot, e2_dot, e3_dot)

            f = x + e_dot * dt

        else:
            rospy.logerr("Not exist this dynamic model (%s)", model)

        # Dynamics derivatives
        Ak = ca.jacobian(f, x)
        Bk = ca.jacobian(f, u)

        # CasADi functions
        self._f = ca.Function('f', [x, u], [f])
        self._A = ca.Function('A_k', [x, u], [Ak])
        self._B = ca.Function('B_k', [x, u], [Bk])

    def f(self, x, u):
        return self._f(x, u)

    def f_prime(self, x, u):
        A_k = self._A(x, u)
        B_k = self._B(x, u)
        return A_k, B_k