import numpy as np
from gym import spaces
from gym.utils import seeding
import random
from scipy.stats import rice
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rlframework import Nav_Agent


class UAV:

    def __init__(self, starting_x, starting_y, starting_z, orientation=np.pi/4, max_speed=4, battery=10,
                 slot_time=1):
        self.x = starting_x
        self.y = starting_y
        self.z = starting_z
        self.status = 1  # 1: Navigation, -1: Data Collection, 0: Finished
        self.orientation = orientation
        self.speed = 0
        self.max_speed = max_speed
        self.battery = battery  # can be assumed it is +infinity
        self.slot_time = slot_time
        # Below parameters are for the energy consumption of the UAV. You can ignore them
        self.P0 = (0.012/8) * 1.225 * 0.05 * 0.79 * pow(400, 3) * pow(0.5, 3)
        self.Pi = (1 + 0.1) * pow(100, 1.5) / np.sqrt(2 * 1.225 * 0.79)
        self.lambda1 = pow(200, 2)
        self.lambda2 = 2 * pow(7.2, 2)
        self.lambda3 = 0.5 * 0.3 * 1.225 * 0.05 * 0.79
        self.trajectory = np.array([np.array([self.x, self.y])])

    def reset_trajectory(self):
        self.trajectory = np.array([np.array([self.x, self.y])])

    def power_consumption(self):  # need modification

        return self.P0 * (1 + 3 * pow(self.speed, 2) / self.lambda1) +\
               self.Pi * np.sqrt(np.sqrt(1 + pow(self.speed, 4)/pow(self.lambda2, 2)) - pow(self.speed, 2)/self.lambda2)+ \
               self.lambda3 * pow(self.speed, 3)

    def hover_consumption(self):  # need modification

        return self.P0 + self.Pi

    def move(self, delta_v, delta_theta):
        x_prev = self.x
        y_prev = self.y
        self.x, self.y, self.speed, self.orientation = self.next_position(delta_v, delta_theta)
        p = self.power_consumption()
        #self.energy -= p * self.slot_time
        return np.array([x_prev, y_prev]), np.array([self.x, self.y]), p  # * self.slot_time
        # check before calling move function if it goes beyond the region borders/collide with obstacles

    def next_position(self, delta_v, delta_theta):
        speed_ = np.clip(self.speed + delta_v, 0, self.max_speed)  # Min Speed = 0?
        orient_ = (self.orientation + delta_theta) % (2 * np.pi)
        x_ = self.x + speed_ * self.slot_time * np.cos(orient_)
        y_ = self.y + speed_ * self.slot_time * np.sin(orient_)
        return x_, y_, speed_, orient_

    def get_internal_state(self, x_lim, y_lim):
        #return np.array([self.x/x_lim, self.y/y_lim, self.speed/self.max_speed, self.orientation/(2 * np.pi)])
        return np.array([self.speed/self.max_speed, self.orientation/(2 * np.pi)])

    def get_location(self):
        return np.array([self.x, self.y, self.z])
    def get_2dlocation(self):
        return np.array([self.x, self.y])


class Sensor:

    def __init__(self, x, y, starting_energy=1, slot_time=1, data=20, t_power=-40):
        self.x = x
        self.y = y
        self.z = 0
        self.battery = starting_energy
        self.slot_time = slot_time
        self.data = data
        self.power = t_power

    def get_location(self):
        return np.array([self.x, self.y, self.z])


class Scheduler:

    def __init__(self, unit_size=50, uav_height=50, obstacle_r=25, max_x=10, max_y=10, n_uav=1, n_sensor=1,
                 n_obstacle=1, max_rangefinder=0.5, num_rangers=3, episode_time_limit=100, target_r=20, max_speed=10,
                 nav_time_slot=1, com_slot_time=1, initial_data=200):

        self.unit_size = unit_size  # Size of each Cell
        self.nav_time_slot = nav_time_slot
        self.com_slot_time = com_slot_time
        self.Mx = max_x  # Number of cells on X axis
        self.My = max_y  # Number of cells on Y axis

        self.n_uav = n_uav  # Number of UAVs
        self.n_sensor = n_sensor  # Number of Sensors
        self.n_obstacle = n_obstacle  # Number of Obstacles

        self.target_radius = target_r
        self.obstacle_radius = obstacle_r  # Obstacle's edge
        self.obs_centers = None
        self.sensor_centers = None
        self.uav_2d_centers = None

        self.max_speed = max_speed
        self.uav_height = uav_height
        self.UAV_blobs = []
        self.UAV_loc_idx = []
        self.grid_map = None

        self.sensor_stats = []
        self.initial_data = initial_data

        self.max_rangefinder = max_rangefinder
        self.num_rangers = num_rangers
        self.ranges = np.ones((n_uav, num_rangers)) * max_rangefinder

        self.state = None
        self.nav_action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.nav_state_space = [9]
        self.nav_agent = Nav_Agent()
        observation = np.zeros(9)
        self.nav_agent.initialize_networks(state=observation.reshape(1, -1))
        self.nav_agent.load_models(number=None)

        self.time_limit = episode_time_limit
        self.time_counter = 0
        self.trajectory = None

        # self.com_agent = Com_Agent()
        # observation = np.zeros(3)
        # self.com_agent.initialize_networks(state=observation.reshape(1, -1))
        # self.com_agent.load_models(number=None)

    def reset(self, random=True, locations=None):

        self.UAV_blobs = []
        self.locate_objects(random, locations)
        self.sensor_stats = self.initial_data * np.ones(self.n_sensor)
        self.UAV_loc_idx = [self.n_sensor for i in range(self.n_uav)]
        #self.current_target = 0
        #self.in_radius = 0
        for uav_id in range(self.n_uav):
            self.set_rangers(uav_id)

        self.time_counter = 0
        #self.trajectory = np.array([self.UAV_blob.get_2dlocation()])

        return self.build_state(-1)

    def locate_objects(self, random, locations):
        if random:
            rnd_numbers = np.random.choice(np.arange(1, self.Mx * self.My),
                                            self.n_obstacle + self.n_sensor, replace=False)
        else:
            rnd_numbers = locations

        Ys = rnd_numbers // self.Mx
        Xs = rnd_numbers % self.Mx
        Ys = np.append(Ys, 1)
        Xs = np.append(Xs, 1)
        self.initialize_gridmap(Ys, Xs)
        Ys_center = self.unit_size * (Ys + 1 / 2)
        Xs_center = self.unit_size * (Xs + 1 / 2)

        self.obs_centers = np.column_stack((Xs_center, Ys_center))[0: self.n_obstacle]
        self.sensor_centers = np.column_stack((Xs_center, Ys_center))[self.n_obstacle:]


        # self.uav_centers = np.array([[0.5,0.5]])    ##UAV's Starting Point
        # self.uav_centers = np.array([0.5, 0.5]) * self.unit_size  # UAV's Starting Point

        Xs_UAVs = self.unit_size * (np.zeros(self.n_uav) + 3/2)
        Ys_UAVs = self.unit_size * (np.zeros(self.n_uav) + 3/2)
        self.uav_2d_centers = np.column_stack((Xs_UAVs, Ys_UAVs))
        for i in range(self.n_uav):
            self.UAV_blobs.append(UAV(self.uav_2d_centers[i][0], self.uav_2d_centers[i][1], self.uav_height, orientation=np.pi/4,
                                max_speed=self.max_speed, slot_time=self.nav_time_slot))

        #self.Sensor_blobs = [Sensor(x=centers[0], y=centers[1]) for centers in self.sensor_centers]
        #print("ENV-Locate Objects OBS SENSOR", self.obs_centers, self.sensor_centers)

    def initialize_gridmap(self, ys, xs):
        self.grid_map = np.zeros((self.My, self.Mx))
        for i in range(0, self.n_obstacle):
            self.grid_map[ys[i], xs[i]] = -1
        for i in range( self.n_obstacle, self.n_obstacle + self.n_sensor):
            self.grid_map[ys[i], xs[i]] = 1
        #print(self.grid_map)

    def set_rangers(self, UAV_id):
        x_uav = self.UAV_blobs[UAV_id].x
        y_uav = self.UAV_blobs[UAV_id].y
        or_uav = self.UAV_blobs[UAV_id].orientation
        angles = np.linspace(or_uav - np.pi / 2, or_uav + np.pi / 2, num=self.num_rangers)
        for i, angle in enumerate(angles):
            self.ranges[UAV_id][i] = self.find_single_range(x_uav, y_uav, angle, self.max_rangefinder)

    def find_single_range(self, x_uav, y_uav, angle, max_r):
        for j in np.arange(1, max_r+1, 1):  # Precision = 1 meter
            x = x_uav + j * np.cos(angle)
            y = y_uav + j * np.sin(angle)
            Nx = int(x//self.unit_size)
            Ny = int(y//self.unit_size)
            if Nx < 0 or Ny < 0 or Nx >= self.Mx or Ny >= self.My:
                return j
            elif self.grid_map[Ny, Nx] == -1:
                Ys_center = self.unit_size * (Ny + 1 / 2)
                Xs_center = self.unit_size * (Nx + 1 / 2)
                if abs(x - Xs_center) <= self.obstacle_radius and abs(y - Ys_center) <= self.obstacle_radius:
                    return j
        return max_r

    def nav_build_state(self, UAV_id, sensor_id):             # Normalized State
        a = (self.sensor_centers[sensor_id] - self.UAV_blobs[UAV_id].get_2dlocation()) / (self.unit_size * np.array([self.Mx, self.My]))
        b = self.UAV_blobs[UAV_id].get_internal_state(self.Mx*self.unit_size, self.My*self.unit_size)
        return np.concatenate((a, b, self.ranges[UAV_id] / self.max_rangefinder), axis=None)

    def build_state(self, UAV_id):
        if UAV_id == -1:   # Single Agent
            # print("Single Agent Solver")
            current_location = \
                [self.UAV_blobs[i].get_2dlocation() for i in range(self.n_uav)] / \
                (self.unit_size * np.array([self.Mx, self.My]))
            return np.concatenate((current_location, self.sensor_stats / self.initial_data), axis=None)

        else:
            current_location = self.UAV_blobs[UAV_id].get_2dlocation() / (self.unit_size * np.array([self.Mx, self.My]))
            return np.concatenate((current_location, self.sensor_stats/self.initial_data), axis=None)

    def step(self, actions):  # actions[i] shows the target sensor for agent i

        nav_penalty = np.zeros(self.n_uav)
        flyToSensor = np.zeros(self.n_uav)
        flyToOrigin = np.zeros(self.n_uav)
        hovering_penalty = np.zeros(self.n_uav)
        dataCollection = np.zeros(self.n_uav)
        finish = np.zeros(self.n_uav)
        #rewards = np.zeros(self.n_uav)

        done = False

        #while True:
        for i in range(self.n_uav):
            if self.UAV_blobs[i].status == 1:   # UAV i is in Navigation Status

                if self.UAV_loc_idx[i] == actions[i]:   # UAV i is already in the Target Location
                    if self.UAV_loc_idx[i] != self.n_sensor:    # is NOT at the Initial Location
                        hovering_penalty[i] += -0.001 * self.UAV_blobs[i].power_consumption()
                    elif not (self.sensor_stats == np.zeros(self.n_sensor)).all():  # All the data is not collected
                        hovering_penalty[i] += -1
                    continue
                else:                                   # UAV i is moving towards a Sensor or Initial Location
                    self.update_nav(actions, i)
                    nav_penalty[i] += 0.001 * -self.UAV_blobs[i].power_consumption()

                    distance = np.linalg.norm(self.UAV_blobs[i].get_2dlocation() - self.sensor_centers[actions[i]])
                    if distance < self.target_radius:       # UAV has reached the target
                        self.UAV_loc_idx[i] = actions[i]
                        self.UAV_blobs[i].speed = 0
                        if self.UAV_loc_idx[i] != self.n_sensor and self.sensor_stats[actions[i]]:
                            # UAV i has reached a sensor (NOT the Initial Location)
                            self.sensor_stats[actions[i]] = 0  # All the data is collected
                            #dataCollection[i] += 10
                    else:
                        if actions[i] != self.n_sensor:
                            if self.sensor_stats[actions[i]]:
                                flyToSensor[i] += 1
                        elif (self.sensor_stats == np.zeros(self.n_sensor)).all():
                            # ALL the data has been collected
                            flyToOrigin[i] += 1
                        self.UAV_loc_idx[i] = -1

            elif self.UAV_blobs[i].status == -1:
                print("UAV is Collecting Data")
            else:
                print("UAV is done")
        # if not (self.UAV_loc_idx == -1 * np.ones(self.n_uav)).all():  # An agent is at the sensor location
        #     break

        if (self.sensor_stats == np.zeros(self.n_sensor)).all() and \
                (self.UAV_loc_idx == self.n_sensor * np.ones(self.n_uav)).all():
            # If all the data is collected, and the UAVs are at the INITIAL location
            done = True
            for i in range(self.n_uav):
                finish[i] += 15
        rewards = nav_penalty + flyToSensor + flyToOrigin + hovering_penalty + dataCollection + finish
        state = self.build_state(UAV_id=-1)
        print("STATES", state)
        print("ACTIONS", actions)
        print("REWARDS", rewards)
        print("Nav Penalty", nav_penalty)
        print("Fly To Uncovered Sensor", flyToSensor)
        print("Fly to Origin after Data Collection", flyToOrigin)
        print("Hovering Penalty", hovering_penalty)
        print("DATA Collection", dataCollection)
        print("Finish", finish)
        print("UAV LOC IDX", self.UAV_loc_idx)

        return state, rewards, done  # Single Agent States

    def update_nav(self, actions, UAV_id):
        observation = self.nav_build_state(UAV_id, sensor_id=actions[UAV_id])
        nav_action = self.nav_agent.choose_action(observation)  # The Observation of UAV i
        nav_action = nav_action.numpy()
        delta_speed = nav_action[0] * self.UAV_blobs[UAV_id].max_speed  # Scale the output of the neural network
        delta_phi = nav_action[1] * np.pi
        prev_pos, new_pos, energy_consumption = self.UAV_blobs[UAV_id].move(delta_speed, delta_phi)
        self.uav_2d_centers[UAV_id] = new_pos
        self.set_rangers(UAV_id)
        self.UAV_blobs[UAV_id].trajectory = np.vstack((self.UAV_blobs[UAV_id].trajectory,
                                                       self.UAV_blobs[UAV_id].get_2dlocation()))

    def get_trajectories(self, episode):
        trajectories = [self.UAV_blobs[0].trajectory]
        for i in range(1, self.n_uav):
            trajectories.append(self.UAV_blobs[i].trajectory)
        self.plot_trajectory(trajectories, self.sensor_centers, self.obs_centers, self.obstacle_radius,
                             self.target_radius, episode)

    def plot_trajectory(self, trajectory, sensors, obstacle, obs_r, target_r, episode):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ['--bo', '--yo']
        for i in range(self.n_uav):
            ax.plot(trajectory[i][:, 0], trajectory[i][:, 1], colors[i], label=f'UAV {i+1}')
        for i in range(self.n_sensor):
            # rect = patches.Rectangle((sensor[0]-obs_radius, sensor[1]-obs_radius), 2*obs_radius, 2*obs_radius,
            #                          linewidth=1, edgecolor='k',  facecolor='g')
            # ax.add_patch(rect)
            sensor = sensors[i]
            ax.scatter(sensor[0], sensor[1], s=100, c='g')
            circle = plt.Circle((sensor[0], sensor[1]), target_r, fill=False, edgecolor='g', linestyle='--')
            ax.add_artist(circle)

        circle = plt.Circle((37.5, 37.5), target_r, fill=False, edgecolor='black', linestyle='--')
        ax.add_artist(circle)
        ax.scatter(37.5, 37.5, s=100, c='black')

        for obs in obstacle:
            rect = patches.Rectangle((obs[0] - obs_r, obs[1] - obs_r), 2*obs_r, 2*obs_r,
                                     linewidth=1, edgecolor='k', facecolor='r')
            ax.add_patch(rect)
        ax.set_aspect('equal', adjustable='box')
        #ax.set_title(f"EPISODE {episode}")
        ax.set_xlim(xmin=-0.0, xmax=self.Mx * self.unit_size)
        ax.set_ylim(ymin=-0.0, ymax=self.My * self.unit_size)
        leg = ax.legend()