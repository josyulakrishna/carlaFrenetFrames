# import pdb
import logging
import random

import carla
import time

import numpy as np

from trajectory_planner import TrajectoryPlanner
# from PythonAPI.carla.agents.navigation.controller import VehiclePIDController
from low_level_control_pid import VehiclePIDController
from low_level_control_stanely import VehicleStanleyController
from trajectory_planner import Draw
# from MotionPlanning.Control.MPC_XY_Frame import *
from MotionPlanning.Control.MPC_XY_Frame_NonLinear import *
# from frenet_planner import FrenetPlanner
from PythonRobotics.PathPlanning.FrenetOptimalTrajectory.frenet_optimal_trajectory import *

class spawnVehicle:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.change_world_set_synchronous()
        self.vehicle = self.spawn_vehicle()
        self.change_camera()

    def change_world_set_synchronous(self):
        self.client.load_world('Town04')
        # self.client.load_world('Town01')
        # time.sleep(2)
        # self.client.reload_world()
        world = self.client.get_world()
        world.unload_map_layer(carla.MapLayer.All)
        world.unload_map_layer(carla.MapLayer.Buildings)
        settings = world.get_settings()
        # fixed_delta_seconds <= max_substep_delta_time * max_substeps.
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 14
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        print("Syncronous mode set")
        world.apply_settings(settings)
        # self.client.get_world().wait_for_tick()
        # world.tick()
        print("World set")

    def spawn_vehicle(self):
        actor_list = self.client.get_world().get_actors()

        for actor in actor_list:
            if actor.type_id == "vehicle.tesla.model3":
                actor.destroy()
        vehicle_blueprint = self.client.get_world().get_blueprint_library().filter('model3')[0]
        print("Spawning Vehicle")
        spwanpoints = self.client.get_world().get_map().get_spawn_points()
        spawn_point = spwanpoints[0]  #[random.randint(0,len(spwanpoints))]
        vehicle = self.client.get_world().spawn_actor(vehicle_blueprint, spawn_point)
        time.sleep(3)
        print("Vehicle Spawned")

        self.actor_id = vehicle.id
        #install collision sensor on the vehicle
        collision_bp = self.client.get_world().get_blueprint_library().filter("sensor.other.collision")[0]
        collision_sensor_transform = carla.Transform(carla.Location(x=0, y=0, z=1))
        collision_sensor = self.client.get_world().spawn_actor(collision_bp, collision_sensor_transform, attach_to=vehicle)
        self.collision_sensor = collision_sensor
        self.collision_sensor.listen(lambda event: self.collision_data(event))
        return vehicle

    def collision_data(self, event):
        actor_we_collide_against = event.other_actor
        print("We collided with {}".format(actor_we_collide_against.type_id))

    def change_camera(self):
        spectator = self.client.get_world().get_spectator()
        v_transform = self.vehicle.get_transform()
        #lift spectator
        v_transform.location.x -= 4
        v_transform.location.z += 15
        v_transform.rotation.pitch = -65
        spectator.set_transform(v_transform)
        # time.sleep(5)
        print("Camera set")



    def get_way_points(self):
        #get lateral length of vehicle
        # vehicle_length = self.vehicle.bounding_box.extent.x
        # #get longitudinal length of vehicle
        # vehicle_width = self.vehicle.bounding_box.extent.y
        waypoints = self.vehicle.get_world().get_map().get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(
        carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Bidirectional)).next(0.01)
        wayi = waypoints[-1]
        for i in range(50):
            waypoints += wayi.next(0.5)
            wayi = waypoints[-1]
        return waypoints

    def draw_points(self, frenet_paths):
        for frenet_path in frenet_paths:
            for i in range(len(frenet_path.x)-1):
                self.client.get_world().debug.draw_point(carla.Location(x=frenet_path.x[i], y=frenet_path.y[i], z=0.01), size=0.1, color=carla.Color(r=235,g=0,b=123,a=10), life_time=1000)

    def draw_frenet_path(self, frenet_paths):
        svw = self.client.get_world()
        for frenet_path in frenet_paths:
            for i in range(len(frenet_path.x)-1):
                svw.debug.draw_arrow(carla.Location(x=frenet_path.x[i], y=frenet_path.y[i], z=0.5), carla.Location(x=frenet_path.x[i+1], y=frenet_path.y[i+1], z=0.5), thickness=0.1, color=carla.Color(r=255,g=0,b=0,a=0), life_time=1000)


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

def move_vehicles(vehicles_list, traffic_manager, target_speed):
    for vehicle_id in vehicles_list:
        vehicle = traffic_manager.get_actor(vehicle_id)
        traffic_manager.auto_lane_change(vehicle.id, False)
        traffic_manager.ignore_lights_percentage(vehicle.id, 100)
        traffic_manager.set_global_distance_to_leading_vehicle(vehicle.id, 2)
        traffic_manager.set_vehicle_max_speed(vehicle.id, target_speed)

        control = carla.VehicleControl(throttle=1.0, steer=0.0)
        vehicle.apply_control(control)

def introduce_traffic(sv, numVehicles, spawn_points):

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    #generate 4 vehicles
    world= sv.get_world()
    vehicles_bps = get_actor_blueprints(world, "vehicle.*", "all")
    blueprints = [x for x in vehicles_bps if int(x.get_attribute('number_of_wheels')) == 4]
    spawn_points = sv.get_world().get_map().get_spawn_points()

    batch = []
    vehicles_list = []
    traffic_manager = sv.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    for n, waypoint in enumerate(spawn_points):
        if n >= numVehicles:
            break
        blueprint = np.random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = np.random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
            # blueprint.set_attribute('hero', 'false')
        blueprint.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(blueprint, waypoint).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in sv.apply_batch_sync(batch, True):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    #move the vehicles to the random locations

    # for vehicle_id in vehicles_list:
    #     vehicle = sv.get_world().get_actor(vehicle_id)
        # traffic_manager.auto_lane_change(vehicle.id, False)
        # traffic_manager.ignore_lights_percentage(vehicle.id, 100)
        # traffic_manager.set_global_distance_to_leading_vehicle(vehicle.id, 2)

        # control = carla.VehicleControl(throttle=1.0, steer=0.0)
        # vehicle.apply_control(control)
    # move_vehicles(vehicles_list, traffic_manager, target_speed=2)


if __name__=="__main__":
    sv = spawnVehicle()
    waypoints = sv.get_way_points()
    draw = Draw(sv.client.get_world())
    draw.draw_waypoints(waypoints)

    # motionPlanner = FrenetPlanner()
    ego_state = [sv.vehicle.get_location().x, sv.vehicle.get_location().y, 0, 0, 0, 0, 0]
    # fpath, lanechange, off_the_road = motionPlanner.run_step_single_path(ego_state, 0, df_n=0, Tf=5, Vf_n=-1)
    # wps_to_go = len(fpath.t) - 3  # -2 bc len gives # of items not the idx of last item + 2wp controller is used
    # f_idx = 1
    wx = [waypoint.transform.location.x for waypoint in waypoints]
    wy = [waypoint.transform.location.y for waypoint in waypoints]
    client = carla.Client('localhost', 2000)
    # introduce_traffic(client, 1, [] )

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    draw.draw_csp_with_arrow(tx, ty, tyaw, life_time=1000)

   #draw lane width
    nv = -(waypoints[0].transform.location.x - waypoints[1].transform.location.x)/(waypoints[0].transform.location.y - waypoints[1].transform.location.y)

    (b, d, a, c) = (
    waypoints[0].transform.location.y, waypoints[1].transform.location.y, waypoints[0].transform.location.x,
    waypoints[1].transform.location.x)
    newdir = ((d - b), (c - a))
    newdir = newdir / np.linalg.norm(newdir)
    newPointLanewidth = newdir*waypoints[0].lane_width
    svw = sv.client.get_world()
    svw.debug.draw_point(carla.Location(x=waypoints[0].transform.location.x+ newPointLanewidth[0], y=waypoints[0].transform.location.y+newPointLanewidth[1], z=waypoints[0].transform.location.z)) #right
    svw.debug.draw_point(carla.Location(x=waypoints[0].transform.location.x - newPointLanewidth[0], y=waypoints[0].transform.location.y - newPointLanewidth[1], z=waypoints[0].transform.location.z)) #left

    # traj = TrajectoryPlanner(sv.client.get_world(), sv.vehicle)
    # traj.generate_target_course([waypoint.transform.location.x for waypoint in waypoints], [waypoint.transform.location.y for waypoint in waypoints])
    # dt = 1.0 / 20.0
    # custom_controller = VehiclePIDController(sv.vehicle, args_lateral={'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': dt},
    #                                          args_longitudinal={'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': dt})
    custom_controller = VehiclePIDController(sv.vehicle)
    # pdb.set_trace()

    # # custom_controller = VehicleStanleyController(sv.vehicle)
    ticks_to_track = len(waypoints)
    ### controller test
    # for i in range(0, ticks_to_track):
    while(np.linalg.norm([sv.vehicle.get_location().x - waypoints[-1].transform.location.x, sv.vehicle.get_location().y - waypoints[-1].transform.location.y]) > 2):
        i = np.min([np.argmin([np.linalg.norm([sv.vehicle.get_location().x - waypoints[i].transform.location.x, sv.vehicle.get_location().y - waypoints[i].transform.location.y]) for i in range(0, ticks_to_track)]) + 1, ticks_to_track-1])
        control_signal = custom_controller.run_step(5, [waypoints[i].transform.location.x, waypoints[i].transform.location.y])
        # wp_prev = [waypoints[i-1].transform.location.x, waypoints[i-1].transform.location.y]
        # wp_curr = [waypoints[i].transform.location.x, waypoints[i].transform.location.y]
        # wp_curr = [sv.vehicle.get_location().x, sv.vehicle.get_location().y]
        # control_signal = custom_controller.run_step(wp_prev, wp_curr)
        # control_signal = custom_controller.run_step_2_wp(5, wp_prev, wp_curr)
        sv.vehicle.apply_control(control_signal)
        sv.client.get_world().tick()
        sv.change_camera()
    #     time.sleep(0.1)
    # client = carla.Client('localhost', 2000)
    # vehicle = spawn_vehicle(client)
    # spectator = client.get_world().get_spectator()
    # v_transform = vehicle.get_transform()
    # #lift spectator
    # v_transform.location.x -= 4
    # v_transform.location.z += 8
    # v_transform.rotation.pitch = -65
    # spectator.set_transform(v_transform)
    # time.sleep(3)
    # # change_world_set_synchronous(client)
    # waypoints = get_way_points(vehicle)
    # draw_waypoints(client.get_world(), waypoints)
    # tx, ty, tyaw, tcur, csp = generate_target_course([waypoint.transform.location.x for waypoint in waypoints[0::15]], [waypoint.transform.location.y for waypoint in waypoints[0::15]])
    # draw_csp_with_arrow(client.get_world(), tx, ty, tyaw)
    # print("Waypoints collected")
    #
    # cx, cy, cyaw, ck, s = cs.calc_spline_course([waypoint.transform.location.x for waypoint in waypoints],
    #                                             [waypoint.transform.location.y for waypoint in waypoints], ds=P.d_dist)
    # sp = calc_speed_profile(cx, cy, cyaw, P.target_speed)
    # ref_path = PATH(cx, cy, cyaw, ck)
    # node = Node(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
    #
    # t1 = 0.0
    # x = [node.x]
    # y = [node.y]
    # yaw = [node.yaw]
    # v = [node.v]
    # t = [0.0]
    # d = [0.0]
    # a = [0.0]
    #
    # delta_opt, a_opt = None, None
    # a_exc, delta_exc = 0.0, 0.0
    #
    # # while t1 < P.time_max:
    # while np.linalg.norm([waypoints[-1].transform.location.x - x[-1],
    #                       waypoints[-1].transform.location.y - y[-1]]) > 1:
    #     z_ref, target_ind = calc_ref_trajectory_in_T_step(node, ref_path, sp)
    #
    #     # _non_linear
    #     z0 = [node.x, node.y, node.yaw, node.v, 0]
    #     # z0 = [node.x, node.y,  node.v, node.yaw]
    #     a_opt, delta_opt, x_opt, y_opt, yaw_opt, v_opt = \
    #         linear_mpc_control(z_ref, z0, a_opt, delta_opt)
    #
    #     if delta_opt is not None:
    #         delta_exc, a_exc = delta_opt[0], a_opt[0]
    #
    #     node.update(a_exc, delta_exc, 1.0)
    #     t1 += P.dt
    #
    #     x.append(node.x)
    #     y.append(node.y)
    #     yaw.append(node.yaw)
    #     v.append(node.v)
    #     t.append(t1)
    #     d.append(delta_exc)
    #     a.append(a_exc)
    #
    #     dist = math.hypot(node.x - cx[-1], node.y - cy[-1])
    #
    #     if dist < P.dist_stop and abs(node.v) < P.speed_stop:
    #         break
    #
    #     dy = (node.yaw - yaw[-2]) / (node.v * P.dt)
    #     # _non_linear
    #     steer = rs.pi_2_pi(-math.atan(P.WB * dy))
    #     # steer = rs.pi_2_pi(math.atan(P.WB * dy))
    #     control_signal = custom_controller.run_step(5, [node.x, node.y])
    #     # [waypoints[len(t)].transform.location.x,
    #     #      waypoints[len(t)].transform.location.y])
    #     control = carla.VehicleControl()
    #     control.steer = steer
    #     control.throttle = control_signal.throttle
    #     control.brake = 0.0
    #     control.hand_brake = False
    #     control.manual_gear_shift = False
    #     spectator = sv.client.get_world().get_spectator()
    #     v_transform = sv.vehicle.get_transform()
    #     # lift spectator
    #     v_transform.location.x -= 4
    #     v_transform.location.z += 8
    #     v_transform.rotation.pitch = -65
    #     spectator.set_transform(v_transform)
    #     sv.vehicle.apply_control(control)
    #     sv.client.get_world().tick()


# def mpc_xy():
    #MPC X-Y Frame
    # cx, cy, cyaw, ck, s = cs.calc_spline_course([waypoint.transform.location.x for waypoint in waypoints], [waypoint.transform.location.y for waypoint in waypoints], ds=P.d_dist)
    # sp = calc_speed_profile(cx, cy, cyaw, P.target_speed)
    # ref_path = PATH(cx, cy, cyaw, ck)
    # node = Node(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
    #
    # t1 = 0.0
    # x = [node.x]
    # y = [node.y]
    # yaw = [node.yaw]
    # v = [node.v]
    # t = [0.0]
    # d = [0.0]
    # a = [0.0]
    #
    # delta_opt, a_opt = None, None
    # a_exc, delta_exc = 0.0, 0.0
    #
    # # while t1 < P.time_max:
    # while np.linalg.norm([waypoints[-1].transform.location.x - x[-1],
    #                       waypoints[-1].transform.location.y - y[-1]]) > 1:
    #     z_ref, target_ind = calc_ref_trajectory_in_T_step(node, ref_path, sp)
    #
    # # _non_linear
    #     z0 = [node.x, node.y, node.yaw, node.v, 0]
    #     # z0 = [node.x, node.y,  node.v, node.yaw]
    #     a_opt, delta_opt, x_opt, y_opt, yaw_opt, v_opt = \
    #         linear_mpc_control(z_ref, z0, a_opt, delta_opt)
    #
    #     if delta_opt is not None:
    #         delta_exc, a_exc = delta_opt[0], a_opt[0]
    #
    #     node.update(a_exc, delta_exc, 1.0)
    #     t1 += P.dt
    #
    #     x.append(node.x)
    #     y.append(node.y)
    #     yaw.append(node.yaw)
    #     v.append(node.v)
    #     t.append(t1)
    #     d.append(delta_exc)
    #     a.append(a_exc)
    #
    #     dist = math.hypot(node.x - cx[-1], node.y - cy[-1])
    #
    #     if dist < P.dist_stop and abs(node.v) < P.speed_stop:
    #         break
    #
    #     dy = (node.yaw - yaw[-2]) / (node.v * P.dt)
    #     # _non_linear
    #     steer = rs.pi_2_pi(-math.atan(P.WB * dy))
    #     # steer = rs.pi_2_pi(math.atan(P.WB * dy))
    #     control_signal = custom_controller.run_step(5, [node.x, node.y])
    #                                                 # [waypoints[len(t)].transform.location.x,
    #                                                 #      waypoints[len(t)].transform.location.y])
    #     control = carla.VehicleControl()
    #     control.steer = steer
    #     control.throttle = control_signal.throttle
    #     control.brake = 0.0
    #     control.hand_brake = False
    #     control.manual_gear_shift = False
    #     spectator = sv.client.get_world().get_spectator()
    #     v_transform = sv.vehicle.get_transform()
    #     #lift spectator
    #     v_transform.location.x -= 4
    #     v_transform.location.z += 8
    #     v_transform.rotation.pitch = -65
    #     spectator.set_transform(v_transform)
    #     sv.vehicle.apply_control(control)
    #     sv.client.get_world().tick()
    #     # time.sleep(0.2)
