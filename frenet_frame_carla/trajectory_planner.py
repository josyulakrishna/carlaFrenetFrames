import math
import numpy as np
import numpy.matlib as npm
import time
import cubic_spline_planner
import carla
import matplotlib.pyplot as plt
class Draw:
    def __init__(self, world):
        self.world = world

    def draw_waypoints(self, waypoints, life_time=1000.0):
        for waypoint in waypoints:
            self.world.debug.draw_string(waypoint.transform.location,
                                            'O',
                                            draw_shadow=False,
                                            color=carla.Color(r=0, g=255, b=0),
                                            life_time=life_time,
                                            persistent_lines=True
                                         )

    def draw_csp_with_arrow(self, tx, ty, tyaw, life_time=50.0):
        for i in range(len(tx)-1):
            start_loc = carla.Location(x=tx[i], y=ty[i], z=0.5)
            end_loc = carla.Location(x=tx[i+1], y=ty[i+1], z=0.5)
            # rot = carla.Rotation(pitch=0, yaw=math.degrees(tyaw[i]), roll=0)
            self.world.debug.draw_arrow(start_loc, end_loc, life_time=life_time)


class TrajectoryPlanner:
    def __init__(self, world, vehicle, target_speed=20):
        self.world = world
        self.vehicle = vehicle
        self.draw = Draw(world)

    def generate_target_course(self, x, y):
        csp = cubic_spline_planner.Spline2D(x, y)
        s = np.arange(0, csp.s[-1], 0.1)
        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = csp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(csp.calc_yaw(i_s))
            rk.append(csp.calc_curvature(i_s))

        self.draw.draw_csp_with_arrow(rx, ry, ryaw)
        return rx, ry, ryaw, rk, csp
