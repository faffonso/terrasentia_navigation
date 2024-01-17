#!/usr/bin/env python3

import casadi as ca
import numpy as np

class NMPC:
    def __init__(self, dt=0.1, N=10, Q_x=1.0, Q_y=1.0, Q_theta=1.0, R_v=1.0, R_omega=1.0):
        self.dt = dt
        self.N  = N

        self.Q = np.diag([Q_x, Q_y, Q_theta])
        self.R = np.diag([R_v, R_omega])

        # Optimization struct
        opti = ca.Opti()

        # State space
        opt_states = opti.variable(N+1, 3)
        self.opt_states = opt_states

        x       = opt_states[:, 0]
        y       = opt_states[:, 1]
        theta   = opt_states[:, 2]

        opt_controls = opti.variable(N, 2)
        self.opt_controls = opt_controls

        v       = opt_controls[:, 0]
        omega   = opt_controls[:, 1]

        # Parameters
        self.opt_x0 = opti.parameter(3)
        self.opt_xref = opti.parameter(3)

        # Init condition
        x0 = self.opt_x0.T
        opti.subject_to(opt_states[0, :] == x0)

        # Subject to dynamic system
        for k in range(N):
            xs = opt_states[k, :]
            us = opt_controls[k, :]

            x_next = xs + self.f(xs, us).T * dt
            opti.subject_to(opt_states[k+1, :] == x_next)
    
        # Cost functions
        obj = 0
        xref = self.opt_xref

        for k in range(N):
            xs = opt_states[k, :].T
            us = opt_controls[k, :].T

            obj += self.l((xs - xref), us)

        opti.minimize(obj)

        # Boundrary and control conditions
        v_max       = 4.0
        omega_max   = 1.5

        opti.subject_to(opti.bounded(0, v, v_max))
        opti.subject_to(opti.bounded(-omega_max, omega, omega_max))

        # Solver settings
        opts_setting = {
            'ipopt.max_iter':100, 
            'ipopt.print_level':0, 
            'print_time':0, 
            'ipopt.acceptable_tol':1e-8, 
            'ipopt.acceptable_obj_change_tol':1e-6}

        opti.solver('ipopt', opts_setting)

        self.opti = opti
        print(f'Optimization problem {self.opti}')

    def fit(self, xs, xref, x0, u0):
        self.opti.set_value(self.opt_xref, xref)
        self.opti.set_value(self.opt_x0, x0)

        self.opti.set_initial(self.opt_controls, u0)
        self.opti.set_initial(self.opt_states, xs)

        sol = self.opti.solve()

        u = sol.value(self.opt_controls)
        x = sol.value(self.opt_states)
        return x, u
    
    def f(self, x, u):
        x1 = u[0]*ca.cos(x[2])
        x2 = u[0]*ca.sin(x[2])
        x3 = u[1]

        return ca.vertcat(x1, x2, x3)
    
    def l(self, x, u = 0):
        running_cost = x.T  @ self.Q @ x + u.T @ self.R @ u
        return running_cost