from collections import defaultdict

from PythonRobotics.PathPlanning.FrenetOptimalTrajectory.frenet_optimal_trajectory import QuarticPolynomial, FrenetPath, calc_global_paths, get_frenet_coord

from QuinticPolynomialsPlanner.quintic_polynomials_planner import \
    QuinticPolynomial
from CubicSpline import cubic_spline_planner
from spawn_vehicle import spawnVehicle
import numpy as np
import copy
import carla
import logging

from low_level_control_pid import VehiclePIDController
import matplotlib.pyplot as plt


# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 4  # maximum curvature [1/m]
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


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def introduce_traffic(sv, numVehicles, spawn_points):

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    #generate 4 vehicles
    world= sv.vehicle.get_world()
    vehicles_bps = get_actor_blueprints(world, "vehicle.tesla.model3", "all")
    blueprints = [x for x in vehicles_bps if int(x.get_attribute('number_of_wheels')) == 4]
    spawn_points = sv.vehicle.get_world().get_map().get_spawn_points()

    batch = []
    vehicles_list = []
    traffic_manager = sv.client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(0.5)
    traffic_manager.global_percentage_speed_difference(70.0)
    # spawn_points.transform.location.z += 0.2
    for n, waypoint in enumerate(spawn_points):
        if n >= numVehicles:
            break
        blueprint = np.random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = np.random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(blueprint, waypoint)
                    .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in sv.client.apply_batch_sync(batch, True):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    #move the vehicles to the random locations

    # for v in vehicles_list:
    #     v.go_to_location(sv.client.get_world().get_random_location_from_navigation())

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

def calc_frenet_paths(waypoint, c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []
    width_range = None
    MAX_ROAD_WIDTH = waypoint.lane_width
    if waypoint.lane_change == carla.libcarla.LaneChange.Right:
        width_range = np.arange(0, MAX_ROAD_WIDTH+1, D_ROAD_W)
    elif waypoint.lane_change == carla.libcarla.LaneChange.Left:
        width_range = np.arange(-MAX_ROAD_WIDTH-1,0, D_ROAD_W)
    elif waypoint.lane_change == carla.libcarla.LaneChange.Both:
        MAX_ROAD_WIDTH = 2*waypoint.lane_width
        width_range = np.arange(-MAX_ROAD_WIDTH-1,MAX_ROAD_WIDTH+1, D_ROAD_W)
    else:
        width_range = np.arange(-MAX_ROAD_WIDTH-1,MAX_ROAD_WIDTH+1, D_ROAD_W)

    # generate path to each offset goal

    for di in width_range:

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
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

def check_collision(fp,sv):
    #check if the path collides with any vehicles
    ob  = [] # array of obstacle locations
    #get all actors which are vehicles
    vehicles = sv.client.get_world().get_actors().filter('vehicle.*')
    #get vehicle dimensions
    vehicle_dimensions = vehicles[0].bounding_box.extent
    length = vehicle_dimensions.x*2
    width = vehicle_dimensions.y*2
    r = np.sqrt(length ** 2 + width ** 2) / 2
    ROBOT_RADIUS = r
    #for all vehicles other than the ego vehicle check for collision
    egoid = sv.vehicle.id
    #get the obstacle locations
    for v in vehicles:
        if v.id != egoid:
            ob.append(np.array([v.get_location().x,v.get_location().y]))
    ob = np.array(ob)
    ob = ob.reshape(-1, 2)
    #check for collision
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, sv):
    ok_ind = []
    for i, _ in enumerate(fplist):
        # if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
        #     continue
        # # elif any([abs(a) > MAX_ACCEL for a in
        #           fplist[i].s_dd]):  # Max accel check
        #     continue
        # #elif any([abs(c) > MAX_CURVATURE for c in
        #           fplist[i].c]):  # Max curvature check
        #     continue
        if not check_collision(fplist[i], sv):
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, sv, waypoint):
    fplist = calc_frenet_paths(waypoint, c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, sv)
    # sv.draw_frenet_path(fplist)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp
    sv.draw_frenet_path([best_path])

    return best_path

def PID_test():
    sv = spawnVehicle()
    sv.client.get_world().tick()
    #get the waypoints from the map
    waypoints = sv.get_way_points()
    wx = [waypoint.transform.location.x for waypoint in waypoints]
    wy = [waypoint.transform.location.y for waypoint in waypoints]
    ticks_to_track = len(waypoints)
    errors = defaultdict(list)
    # for speed in [2,5,10]:
    speed = 2
    custom_controller = VehiclePIDController(sv.vehicle)
    while (np.linalg.norm([sv.vehicle.get_location().x - waypoints[-1].transform.location.x,
                           sv.vehicle.get_location().y - waypoints[-1].transform.location.y]) > 2):
        i = np.min([np.argmin([np.linalg.norm([sv.vehicle.get_location().x - waypoints[i].transform.location.x,
                                               sv.vehicle.get_location().y - waypoints[i].transform.location.y]) for i
                               in range(0, ticks_to_track)]) + 1, ticks_to_track - 1])
        control_signal = custom_controller.run_step(speed, [waypoints[i].transform.location.x, waypoints[i].transform.location.y])
        sv.vehicle.apply_control(control_signal)
        sv.client.get_world().tick()
        sv.change_camera()
    #plot long and lat errors for PID controller
    fig, ax = plt.subplots(2, 1)
    # ax[0].plot( [5]*len(custom_controller._lon_controller.long_error))
    ax[0].plot(custom_controller._lon_controller.long_error)
    ax[0].set_title('Longitudinal Error')

    ax[1].plot(custom_controller._lat_controller.lat_error)
    ax[1].set_title('Lateral Error')
    plt.show()
    np.save('long_error_{0}'.format(speed), np.array(custom_controller._lon_controller.long_error))
    np.save('lat_error_{0}'.format(speed), np.array(custom_controller._lat_controller.lat_error))
    print("done")

def plot_errors():
    fig, ax = plt.subplots(2, 1)
    # plt.title('PID Error mean curvature 0.052')
    fig.supxlabel('Time')
    fig.supylabel('Error')
    fig.suptitle('PID Error mean curvature 0.002')
    for speed in [2,5,10]:
        ax[0].plot(np.load('long_error_{0}.npy'.format(speed)), label='speed {0}'.format(speed))
        ax[0].set_title('Longitudinal Error')

        ax[1].plot(np.load('lat_error_{0}.npy'.format(speed)), label='speed {0}'.format(speed))
        ax[1].set_title('Lateral Error')

    ax[0].legend()
    # ax[1].legend()
    plt.show()

def testPathGeneration():
    #first spawn a vehicle
    sv = spawnVehicle()
    sv.client.get_world().tick()
    #get the waypoints from the map
    waypoints = sv.get_way_points()
    wx = [waypoint.transform.location.x for waypoint in waypoints]
    wy = [waypoint.transform.location.y for waypoint in waypoints]

    #introduce traffic to 30m ahead of the ego vehicle
    # introduce_traffic(sv, 40, [waypoints[49]])

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    #dummy initial states

    c_speed = 0  # current speed [m/s]
    c_accel = 0.0  # current acceleration [m/ss]
    c_d = 0.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position

    #call the frenet path generation function
    # fplist = calc_frenet_paths(waypoints[0], c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    # fplist = calc_global_paths(fplist, csp)
    #
    # sv.draw_frenet_path(fplist)

    #check the velocity of the first path
    #spawn in front of the vehicle
    vehicle_blueprint = sv.client.get_world().get_blueprint_library().filter('model3')[0]
    #yaw = angle between two waypoints in radians
    yaw = np.arctan2(waypoints[30].transform.location.y, waypoints[30].transform.location.x)
    # vehicle = sv.client.get_world().spawn_actor(vehicle_blueprint, carla.Transform(carla.Location(x=tx[30], y=ty[30], z=0.2), carla.Rotation()))
    sv.client.get_world().spawn_actor(vehicle_blueprint, carla.Transform(
        carla.Location(x=waypoints[30].transform.location.x, y=waypoints[30].transform.location.y, z=2),
        carla.Rotation(yaw=yaw, pitch=waypoints[0].transform.rotation.pitch, roll=waypoints[0].transform.rotation.roll)))

    wpendx = waypoints[-1].transform.location.x
    wpendy = waypoints[-1].transform.location.y

    while True:
        if np.linalg.norm([sv.vehicle.get_location().x-wpendx,sv.vehicle.get_location().y-wpendy]) >= 6:
            path = frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, sv, waypoints[0])
        else:
            waypoints = sv.get_way_points()
            wx = [waypoint.transform.location.x for waypoint in waypoints]
            wy = [waypoint.transform.location.y for waypoint in waypoints]
            tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
            wpendx = waypoints[-1].transform.location.x
            wpendy = waypoints[-1].transform.location.y
            s0=0
            path = frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, sv, waypoints[0])
        #vehicle coordinates
        x = sv.vehicle.get_location().x
        y = sv.vehicle.get_location().y
        s0, c_d = get_frenet_coord(x, y, list(zip(csp.sx.y, csp.sy.y)))
        print("s0 gf: ", s0)
        print("c_d gf: ", c_d," d ", path.d[1])
        # s0 = path.s[1]
        # c_d = path.d[1]
        print("path x: ", x, " car  ",  path.x[1])
        print("path y: ", y, " car ", path.y[1])
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]

        c_speed = 2 #np.sqrt(sv.vehicle.get_velocity().x**2 + sv.vehicle.get_velocity().y**2)
        c_accel = 0.5 #np.sqrt(sv.vehicle.get_acceleration().x**2 + sv.vehicle.get_acceleration().y**2)

        # c_speed = path.s_d[1]
        # c_accel = path.s_dd[1]
        #move the vehicle along the path with c_speed
        custom_controller = VehiclePIDController(sv.vehicle)
        #get x, y, yaw of the target point on the path
        control_signal = custom_controller.run_step(c_speed, [path.x[1], path.y[1]])
        print("control signal: ", control_signal)
        sv.vehicle.apply_control(control_signal)
        sv.client.get_world().tick()
        sv.change_camera()

if __name__ == '__main__':
    testPathGeneration()
    # PID_test()
    # plot_errors()