import numpy as np
from gym import spaces
from gym.utils import seeding
import random
from scipy.stats import rice
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class UAV:

    def __init__(self, starting_x, starting_y, starting_z, orientation=np.pi/4, max_speed=4, starting_energy=10,
                 slot_time=1):
        self.x = starting_x
        self.y = starting_y
        self.z = starting_z
        self.orientation = orientation
        self.speed = 0
        self.max_speed = max_speed
        self.energy = starting_energy  # can be assumed it is +infinity
        self.slot_time = slot_time

        # Below parameters are for the energy consumption of the UAV. You can ignore them
        self.P0 = (0.012/8) * 1.225 * 0.05 * 0.79 * pow(400, 3) * pow(0.5, 3)
        self.Pi = (1 + 0.1) * pow(100, 1.5) / np.sqrt(2 * 1.225 * 0.79)
        self.lambda1 = pow(200, 2)
        self.lambda2 = 2 * pow(7.2, 2)
        self.lambda3 = 0.5 * 0.3 * 1.225 * 0.05 * 0.79

    def power_consumption(self):  # need modification

        return self.P0 * (1 + 3 * pow(self.speed, 2) / self.lambda1) +\
               self.Pi * np.sqrt(np.sqrt(1 + pow(self.speed, 4)/pow(self.lambda2, 2)) - pow(self.speed, 2)/self.lambda2)+ \
               self.lambda3 * pow(self.speed, 3)

    def move(self, delta_v, delta_theta):
        x_prev = self.x
        y_prev = self.y
        self.x, self.y, self.speed, self.orientation = self.next_position(delta_v, delta_theta) 
        p = self.power_consumption()
        #self.energy -= p * self.slot_time
        return np.array([[x_prev, y_prev]]), np.array([[self.x, self.y]]), p  # * self.slot_time
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
        self.energy = starting_energy
        self.slot_time = slot_time
        self.data = data
        self.power = t_power

    def get_location(self):
        return np.array([self.x, self.y, self.z])


class Nav_Dynamics:

    def __init__(self, unit_size=1., uav_height=10, obstacle_r=0.5, r_max=1., max_x=10, max_y=10, n_uav=1, n_sensor=1,
                 n_obstacle=1, max_rangefinder=0.5, num_rangers=3, episode_time_limit=100, target_r=20, max_speed=10,
                 time_slot=1):
        self.R_MAX = r_max
        self.unit_size = unit_size      # Size of each Cell
        self.time_slot = time_slot
        self.Mx = max_x     # Number of cells on X axis
        self.My = max_y     # Number of cells on Y axis

        self.n_uav = n_uav  # Number of UAVs
        self.n_sensor = n_sensor    # Number of Sensors
        self.n_obstacle = n_obstacle    # Number of Obstacles

        self.target_radius = target_r
        self.obstacle_radius = obstacle_r  # Obstacle's edge
        self.obs_centers = None
        self.sensor_centers = None
        self.uav_2d_centers = None
        self.max_speed = max_speed
        self.uav_height = uav_height
        self.UAV_blob = None
        self.grid_map = None

        self.schedule = None  # the order of targets to be visited

        self.sensor_covered = None
        self.current_target = None

        self.max_rangefinder = max_rangefinder
        self.num_rangers = num_rangers
        self.ranges = np.ones(num_rangers) * max_rangefinder
        self.trans_reward_decay = 0.5

        self.state = None
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.state_space = [9]

        self.time_limit = episode_time_limit
        self.time_counter = 0
        self.trajectory = None

    def reset(self, schedule, random=True, locations=None):
        self.schedule = schedule
        print(locations)
        self.locate_objects(random, locations)

        self.sensor_covered = -1 * np.ones(self.n_sensor)
        self.current_target = 0
        self.in_radius = 0

        self.set_rangers()

        self.state = self.build_state()

        self.time_counter = 0
        self.trajectory = np.array([self.UAV_blob.get_2dlocation()])

        return self.state
        # self.UAV_list = [UAV(center[0],center[1]) for center in self.uav_centers]

    def locate_objects(self, random, locations):
        if random:
            rnd_numbers = np.random.choice(np.arange(0, self.Mx * self.My),
                                           self.n_uav + self.n_obstacle + self.n_sensor, replace=False)
        else:
            rnd_numbers = locations


        Ys = rnd_numbers // self.Mx
        Xs = rnd_numbers % self.Mx

        self.initialize_gridmap(Ys, Xs)
        Ys_center = self.unit_size * (Ys + 1 / 2)
        Xs_center = self.unit_size * (Xs + 1 / 2)

        # self.uav_centers = np.array([0.5, 0.5]) * self.unit_size  # UAV's Starting Point
        self.uav_2d_centers = np.column_stack((Xs_center, Ys_center))[:self.n_uav]
        self.obs_centers = np.column_stack((Xs_center, Ys_center))[self.n_uav:self.n_uav + self.n_obstacle]
        self.sensor_centers = np.column_stack((Xs_center, Ys_center))[self.n_uav + self.n_obstacle:]
        # self.uav_centers = np.array([[0.5,0.5]])    ##UAV's Starting Point
        self.UAV_blob = UAV(self.uav_2d_centers[0][0], self.uav_2d_centers[0][1], self.uav_height, orientation=np.pi/4,
                            max_speed=self.max_speed, slot_time=self.time_slot)
        #self.Sensor_blobs = [Sensor(x=centers[0], y=centers[1]) for centers in self.sensor_centers]
        #print("ENV-Locate Objects OBS SENSOR", self.obs_centers, self.sensor_centers)

    def initialize_gridmap(self, ys, xs):

        #  gridmap of the environment: -1 shows the obstalces, 1 shows the destination

        self.grid_map = np.zeros((self.My, self.Mx))
        for i in range(self.n_uav, self.n_uav + self.n_obstacle):
            self.grid_map[ys[i], xs[i]] = -1
        for i in range(self.n_uav + self.n_obstacle, self.n_uav + self.n_obstacle + self.n_sensor):
            self.grid_map[ys[i], xs[i]] = 1
        print(self.grid_map)

    def build_state(self):             # Normalized State
        # a = (self.sensor_centers - self.uav_centers) / (self.unit_size * np.array([self.Mx, self.My]))
        i = self.schedule[self.choose_target()]
        a = (self.sensor_centers[i] - self.UAV_blob.get_2dlocation()) / (self.unit_size * np.array([self.Mx, self.My]))
        b = self.UAV_blob.get_internal_state(self.Mx*self.unit_size, self.My*self.unit_size)
        self.raw_state = np.array([self.UAV_blob.x, self.UAV_blob.y, self.UAV_blob.speed,
                                   self.UAV_blob.orientation*180/np.pi, self.ranges])
        #return np.concatenate((a, self.sensor_covered, b, self.ranges/self.max_rangefinder), axis=None)
        return np.concatenate((a, b, self.ranges / self.max_rangefinder), axis=None)

    def set_rangers(self): # updates the range finder readings
        x_uav = self.UAV_blob.x
        y_uav = self.UAV_blob.y
        or_uav = self.UAV_blob.orientation
        angles = np.linspace(or_uav - np.pi/2, or_uav + np.pi/2, num=self.num_rangers)
        for i, angle in enumerate(angles):
            self.ranges[i] = self.find_single_range(x_uav, y_uav, angle, self.max_rangefinder)

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

    def step(self, action):

        self.time_counter += 1

        action = action.numpy()
        delta_speed = action[0] * self.UAV_blob.max_speed  # Scale the output of the neural network
        delta_phi = action[1] * np.pi
        print("CURRENT RAW STATE", self.raw_state)
        print("ACTION", delta_speed, delta_phi*180/np.pi)

        # Move the UAV and watch for: 1-collision with obstacles, 2-out the border
        collision, border, sensor_reached, sensor_num = self.detect_event(delta_speed, delta_phi)
        print("Collision", "Border", "Target", collision, border, sensor_reached)

        done = False
        trans_reward = 0
        energy_consumption = 0
        obs_penalty = 0
        border_penalty = 0
        free_reward = 0
        finish_reward = 0
        #print("ENVIRONMENT TIMER", self.time_counter)

        if collision:
            obs_penalty = -20
        elif border:
            border_penalty = -20

        else:

            prev_pos, new_pos, energy_consumption = self.UAV_blob.move(delta_speed, delta_phi)
            trans_reward = self.transition_reward(prev_pos, new_pos, sensor_reached)

            self.set_rangers()
            i = np.argmin(self.ranges)
            obs_penalty = -20 * np.exp(-self.ranges[i]/(0.1*self.max_rangefinder))
            if self.ranges[2] == self.max_rangefinder:  # UAV heading towards a free-obstacle direction
                free_reward = 1

            if sensor_reached:
                self.sensor_covered[sensor_num] = 1
                self.current_target += 1

                done = (self.current_target == len(self.schedule))
                finish_reward = 50

            # target = np.all(self.sensor_covered == np.ones(self.n_sensor))
            # self.uav_centers = new_pos

        new_state = self.build_state()

        # You can remove the energy consumption penalty
        reward = - 0.002 * energy_consumption + obs_penalty + border_penalty + finish_reward + 0.3 * trans_reward

        self.trajectory = np.vstack((self.trajectory, self.UAV_blob.get_2dlocation()))
        print("TOTAL REWARD", reward, "| TRANS Reward: ", 0.1 * trans_reward, "| OBS PEN", obs_penalty,
              "| Energy Penalty", 0.002 * energy_consumption)
        print("RAW NEW STATE", self.raw_state)
        print("NEW STATE ", new_state)
        return new_state, reward, done, sensor_reached, self.trajectory, self.obs_centers, self.sensor_centers,  energy_consumption

    def detect_event(self, delta_v, delta_phi):
        collision = False
        border = False

        x_, y_, speed_, orient_ = self.UAV_blob.next_position(delta_v, delta_phi)
        x = self.UAV_blob.x
        y = self.UAV_blob.y

        if x_ <= 0 or x_ >= self.Mx * self.unit_size \
                or y_ <= 0 or y_ >= self.My * self.unit_size:
            border = True

        obs_dis = self.find_single_range(x, y, orient_, speed_ * self.time_slot)  # it doesnt reach the last element
        if obs_dis != speed_ * self.time_slot:
            collision = True

        i = self.schedule[self.choose_target()]
        distance = np.linalg.norm(np.array([x_, y_]) - self.sensor_centers[i])
        target_reached = 1 if distance < self.target_radius else 0

        return collision, border, target_reached, i

    def transition_reward(self, prev_pos, new_pos, sensor_reached):
        d_uav_sensor_1 = np.linalg.norm(prev_pos - self.sensor_centers, axis=1)
        d_uav_sensor_2 = np.linalg.norm(new_pos - self.sensor_centers, axis=1) * (1-sensor_reached)
        i = self.schedule[self.current_target]

        trans_reward = d_uav_sensor_1[i] - d_uav_sensor_2[i]
        # decay = 1 - np.exp(-d_uav_sensor_1/self.UAV_blob.z)
        #decay = self.in_radius
        return trans_reward  # * (1-decay)

    def choose_target(self):
        if self.current_target < self.n_sensor:
            return self.current_target
        else:
            return self.current_target - 1
        # sorted_d = np.argsort(d_uav_sensor_1)
        # for i in sorted_d:
        #     if self.sensor_covered[i] == 0:
        #         break
        # for i in range(self.n_sensor):
        #     if self.sensor_covered[i] == -1:
        #         self.current_target = i
        #         return self.schedule[i]

    def show_env(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for sensor in self.sensor_centers:
            # rect = patches.Rectangle((sensor[0]-obs_radius, sensor[1]-obs_radius), 2*obs_radius, 2*obs_radius,
            #                          linewidth=1, edgecolor='k',  facecolor='g')
            # ax.add_patch(rect)
            ax.scatter(sensor[0], sensor[1], s=100, c='g')
            circle = plt.Circle((sensor[0], sensor[1]),  self.target_radius, fill=False, edgecolor='g', linestyle='--')
            ax.add_artist(circle)

        for obs in self.obs_centers:
            rect = patches.Rectangle((obs[0] - self.obstacle_radius, obs[1] - self.obstacle_radius),
                                     2*self.obstacle_radius, 2*self.obstacle_radius,
                                     linewidth=1, edgecolor='k', facecolor='r')
            ax.add_patch(rect)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xmin=-0.0, xmax=24 * self.unit_size+0)
        ax.set_ylim(ymin=-0.0, ymax=24 * self.unit_size+0)

