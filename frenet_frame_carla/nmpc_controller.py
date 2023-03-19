#Udacity Kinematic Motion Model, sadly can't use the pacejka tire force based model
import cvxpy as cp
import numpy as np
# x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
# y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
# psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
# v_[t+1] = v[t] + a[t] * dt
# cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
# epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt

class MPCController:
    def __init__(self, xinit, yinit, psiinit, vinit, cteinit, epsiinit, Lf, xref, yref, yawref, dt, N):
        self.cteref = 0.1
        self.cteyawref = 0.1
        self.x = xinit
        self.y = yinit
        self.psi = psiinit
        self.v = vinit
        self.cte = cteinit
        self.epsi = epsiinit
        self.Lf = Lf
        self.dt = dt
        self.N = N
        self.direct = 1
        self.vmin = 0
        self.vmax = 13.89
        self.xref = np.array(xref)
        self.yref = np.array(yref)
        self.yawref = yawref
        self.nx = 4
        self.nu = 2
        self.T = 6


    def motion_model(self, delta, a):
        self.x = self.x + self.v * np.cos(self.psi) * self.dt
        self.y = self.y + self.v * np.sin(self.psi) * self.dt
        self.psi = np.clip(self.psi + (self.v / self.Lf) * np.tan(delta) * self.dt, -np.deg2rad(45), np.deg2rad(45))
        self.v = np.clip(self.v + a * self.dt, self.vmin, self.vmax)

    def nearest_index(self):
        dx = [self.x - icx for icx in self.xref]
        dy = [self.y - icy for icy in self.yref]
        d = np.hypot(dx, dy)
        ind = np.argmin(d)
        return ind, d[ind]

    def calc_ref_trajectory_in_T_step(self):
        """
        calc referent trajectory in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param node: current informationzx
        """

        z_ref = np.zeros((self.nx, self.T + 1))
        length = self.xref.shape[0]

        ind, _ = self.nearest_index()
        self.ind_old = self.ind_old+ind

        z_ref[0, 0] = self.xref[ind]
        z_ref[1, 0] = self.yref[ind]
        z_ref[2, 0] = 5
        z_ref[3, 0] = self.yawref[ind]

        dist_move = 0.0

        for i in range(1, self.T + 1):
            dist_move += abs(self.v) * self.dt
            ind_move = int(round(dist_move))
            index = min(ind + ind_move, length - 1)

            z_ref[0, i] = self.xref[index]
            z_ref[1, i] = self.yref[index]
            z_ref[2, i] = 5
            z_ref[3, i] = self.yawref[index]