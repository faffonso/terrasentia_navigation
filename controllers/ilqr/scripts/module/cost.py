#!/usr/bin/env python3

import casadi as ca
import numpy as np

class Cost():
    def __init__(self, N, 
                 Qf_x, Qf_y, Qf_theta, 
                 Q_x, Q_y, Q_theta,
                 R_v, R_omega):

        # Problem formulation

        self.N = N      # Horizon size

        Qf = np.diag([Qf_x, Qf_y, Qf_theta])
        Q = np.diag([Q_x, Q_y, Q_theta]) 
        R = np.diag([R_v, R_omega]) 

        self.Qf = Qf    # Weight matrix for final state
        self.Q = Q      # Weight matrix for state
        self.R = R      # Weight matrix for action control
        

        x = ca.SX.sym('e', 3)   # State
        u = ca.SX.sym('u', 2)   # Action control

        l = 1/2 * (x.T @ Q @ x + u.T @ R @ u)   # Running cost
        lf = 1/2 * (x.T @ Qf @ x)               # Cost at final state

        # Running cost derivatives

        lx = ca.jacobian(l, x) 
        lu = ca.jacobian(l, u) 
        lxx = ca.jacobian(lx, x)  
        luu = ca.jacobian(lu, u)

        # CasADi functions

        self._l = ca.Function('l', [x, u], [l])
        self._lf = ca.Function('lf', [x], [lf])  
        
        self._lx = ca.Function('lx', [x], [lx])
        self._lu = ca.Function('lu', [u], [lu])
        self._lxx = ca.Function('lxx', [x], [lxx])
        self._luu = ca.Function('luu', [u], [luu])
        
    
    def _trajectory_cost(self, xs, us):
        J = 0
        N = xs.shape[0]

        for n in range (N-1):
            J += self._l(xs[n], us[n])

        J += self._lf(xs[N-1])

        return J

    def l_prime(self, x, u):
        l_x = self.Q @ x
        l_u = self.R @ u

        l_xx = self.Q
        l_uu = self.R
        l_xu = 0

        return l_x, l_u, l_xx, l_xu, l_uu
    
    def get_Qf(self):
        return self.Qf