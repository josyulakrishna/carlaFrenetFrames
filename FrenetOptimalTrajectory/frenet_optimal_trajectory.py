"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
(https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
(https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from QuinticPolynomialsPlanner.quintic_polynomials_planner import \
    QuinticPolynomial
from CubicSpline import cubic_spline_planner

SIM_LOOP = 500

# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 7.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.1  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0

show_animation = True


class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt



def get_closest_waypoints(x, y, total_wps):
    min_len = 1e10
    closest_wp = 0

    for i in range(len(total_wps)):
        dist = get_dist(x, y, total_wps[i][0], total_wps[i][1])

        if dist < min_len:
            min_len = dist
            closest_wp = i

    return closest_wp

def next_waypoint(x, y, total_wps):
    closest_wp = get_closest_waypoints(x, y, total_wps)
    if closest_wp + 1 == len(total_wps):
        return None

    map_vec = [total_wps[closest_wp + 1][0] - total_wps[closest_wp][0],
               total_wps[closest_wp + 1][1] - total_wps[closest_wp][1]]
    ego_vec = [x - total_wps[closest_wp][0], y - total_wps[closest_wp][1]]

    direction  = np.sign(np.dot(map_vec, ego_vec))

    if direction >= 0:
        next_wp = closest_wp + 1
    else:
        next_wp = closest_wp

    return next_wp

def get_dist(x, y, _x, _y):
    return np.sqrt((x - _x)**2 + (y - _y)**2)

def get_frenet_coord(x,y,total_wps):
    next_wp = next_waypoint(x,y,total_wps)
    if not next_wp:
        return 0,0
    prev_wp = next_wp - 1

    n_x = total_wps[next_wp][0] - total_wps[prev_wp][0]
    n_y = total_wps[next_wp][1] - total_wps[prev_wp][1]
    x_x = x - total_wps[prev_wp][0]
    x_y = y - total_wps[prev_wp][1]

    # print "next/prev wps:", next_wp, " ", prev_wp
    # print "wp_next/wp_prev: ", total_wps[next_wp], total_wps[prev_wp]
    proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y)
    proj_x = proj_norm*n_x
    proj_y = proj_norm*n_y

    #-------- get frenet d
    frenet_d = get_dist(x_x,x_y,proj_x,proj_y)

    ego_vec = [x-total_wps[prev_wp][0], y-total_wps[prev_wp][1], 0]
    map_vec = [n_x, n_y, 0]
    d_cross = np.cross(ego_vec,map_vec)
    if d_cross[-1] > 0:
        frenet_d = -frenet_d

    #-------- get frenet s
    frenet_s = 0
    for i in range(prev_wp):
        frenet_s = frenet_s + get_dist(total_wps[i][0],total_wps[i][1],total_wps[i+1][0],total_wps[i+1][1]);

    frenet_s = frenet_s + get_dist(0,0,proj_x,proj_y)

    return frenet_s, frenet_d

def get_cartesian(s,d,total_wps,wp_s):
    prev_wp = 0
    s = np.mod(s, wp_s[-1]) # EDITED
    while (s > wp_s[prev_wp + 1]) and (prev_wp < len(wp_s) - 2):
        prev_wp = prev_wp + 1

    next_wp = np.mod(prev_wp + 1, len(total_wps))
    dx = (total_wps[next_wp][0] - total_wps[prev_wp][0])
    dy = (total_wps[next_wp][1] - total_wps[prev_wp][1])

    heading = np.arctan2(dy, dx)

    seg_s = s - wp_s[prev_wp];

    seg_x = total_wps[prev_wp][0] + seg_s*np.cos(heading);
    seg_y = total_wps[prev_wp][1] + seg_s*np.sin(heading);

    perp_heading = heading + 90 * np.pi/180;
    x = seg_x + d*np.cos(perp_heading);
    y = seg_y + d*np.sin(perp_heading);

    return x,y,heading

class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

        # self.csp = copy.deepcopy(csp)

def calc_frenet_paths(csp, c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()
            # fp.csp = copy.deepcopy(csp)
            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            #  xs, vxs, axs, xe, vxe, axe, time
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            continue
        elif not check_collision(fplist[i], ob):
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths(csp, c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path


def generate_target_course(x, y):
    csp = cubic_spline_planner.CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def main():
    print(__file__ + " start!!")

    # way points
    #  [0.0, 10.0, 20.5, 35.0, 70.5]
    wx = [225.32388305664062,      225.83963012695312,      226.3553924560547,      226.8711395263672,      227.38687133789062,      227.9026336669922,      228.4183807373047,      228.93414306640625,      229.44989013671875,      229.96102905273438,      230.46102905273438,      230.9610137939453,      231.4610137939453,      231.96099853515625,      232.46099853515625,      232.96099853515625,      233.4609832763672,      233.9609832763672,      234.46096801757812,      234.96096801757812,      235.46095275878906,      235.96095275878906,      236.46095275878906,      236.9609375,      237.4609375,      237.96092224121094,      238.46092224121094,      238.96090698242188,      239.46090698242188,      239.96090698242188,      240.4608917236328,      240.9608917236328,      241.46087646484375,      241.96087646484375,      242.4608612060547,      242.9608612060547,      243.46084594726562,      243.96084594726562,      244.46084594726562,      244.96083068847656,      245.46083068847656,      245.9608154296875,      246.4608154296875,      246.96080017089844,      247.46080017089844,      247.96078491210938,      248.46078491210938,      248.9607696533203,      249.4607696533203,      249.9607696533203,      250.46075439453125,      250.91250610351562,      251.3590087890625,      251.80551147460938,      252.2519989013672,      252.69847106933594,      253.1449432373047,      253.59140014648438,      254.0377960205078,      254.4841766357422,      254.9305419921875,      255.37684631347656,      255.8231201171875,      256.2692565917969,      256.7154235839844,      257.1614990234375,      257.6075134277344,      258.0534362792969,      258.4992980957031,      258.945068359375,      259.3907470703125,      259.8363037109375,      260.2818298339844,      260.7272033691406,      261.1724853515625,      261.61761474609375,      262.0626525878906,      262.5075988769531,      262.952392578125,      263.3970031738281,      263.8415222167969,      264.2858581542969,      264.7300720214844,      265.17413330078125,      265.6180114746094,      266.06170654296875,      266.5052795410156,      266.9486389160156,      267.3918762207031,      267.8348388671875,      268.2776794433594,      268.72027587890625,      269.1626892089844,      269.6048889160156,      270.046875,      270.4886779785156,      270.93023681640625,      271.3715515136719,      271.8126525878906,      272.2535095214844,      272.694091796875]     # wy = [0.0, -6.0, 5.0, 6.5, 0.0]
    wy = [-364.1293640136719,      -364.12774658203125,      -364.126708984375,      -364.1261901855469,      -364.12615966796875,      -364.1266174316406,      -364.12762451171875,      -364.129150390625,      -364.1311950683594,      -364.1336669921875,      -364.13623046875,      -364.1388244628906,      -364.1413879394531,      -364.1439514160156,      -364.14654541015625,      -364.14910888671875,      -364.1517028808594,      -364.1542663574219,      -364.1568603515625,      -364.159423828125,      -364.1620178222656,      -364.1645812988281,      -364.1671447753906,      -364.16973876953125,      -364.17230224609375,      -364.1748962402344,      -364.1774597167969,      -364.1800537109375,      -364.1826171875,      -364.1852111816406,      -364.1877746582031,      -364.1903381347656,      -364.19293212890625,      -364.19549560546875,      -364.1980895996094,      -364.2006530761719,      -364.2032470703125,      -364.205810546875,      -364.2084045410156,      -364.2109680175781,      -364.2135314941406,      -364.21612548828125,      -364.21868896484375,      -364.2212829589844,      -364.2238464355469,      -364.2264404296875,      -364.22900390625,      -364.2315979003906,      -364.2341613769531,      -364.2367248535156,      -364.23931884765625,      -364.2409973144531,      -364.2411804199219,      -364.2398376464844,      -364.23699951171875,      -364.2326354980469,      -364.22674560546875,      -364.21929931640625,      -364.21038818359375,      -364.199951171875,      -364.18804931640625,      -364.1745300292969,      -364.1595458984375,      -364.14306640625,      -364.125,      -364.1054992675781,      -364.0844421386719,      -364.0618896484375,      -364.0378112792969,      -364.01220703125,      -363.985107421875,      -363.95648193359375,      -363.9263610839844,      -363.8946838378906,      -363.8615417480469,      -363.8268737792969,      -363.7906799316406,      -363.75299072265625,      -363.7137756347656,      -363.6730651855469,      -363.630859375,      -363.5871276855469,      -363.5418701171875,      -363.4951171875,      -363.4468688964844,      -363.3971252441406,      -363.3457946777344,      -363.29302978515625,      -363.2387390136719,      -363.1829528808594,      -363.12567138671875,      -363.0668640136719,      -363.00653076171875,      -362.94476318359375,      -362.88140869140625,      -362.8166198730469,      -362.7503662109375,      -362.6825256347656,      -362.6131896972656,      -362.54241943359375,      -362.4700927734375]

    # obstacle lists
    ob = np.array([[20.0, 10.0],
                   [30.0, 6.0],
                   [30.0, 8.0],
                   [35.0, 8.0],
                   [50.0, 3.0]
                   ])



    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    # initial state
    c_speed = 10.0 / 3.6  # current speed [m/s]
    c_accel = 0.0  # current acceleration [m/ss]
    c_d = 0.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position

    area = 20.0  # animation area length [m]

    for i in range(SIM_LOOP):
        # make obstacles oscillate in the y direction with a period of 10 seconds
        ob[i%len(ob), 1] = ob[i%len(ob), 1] - 0.5 * math.sin(i / 10.0)

        print("ob", ob)
        path = frenet_optimal_planning(
            csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob)

        #get frenet state
        s, d = get_frenet_coord(path.x[1], path.y[1], zip(path.x, path.y))
        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]
        c_accel = path.s_dd[1]

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(tx, ty)
            plt.plot(ob[:, 0], ob[:, 1], "xk")
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
            plt.grid(True)
            plt.pause(0.0001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()
