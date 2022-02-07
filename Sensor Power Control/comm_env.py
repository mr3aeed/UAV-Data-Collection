import numpy as np
from gym import spaces
import random
from scipy.stats import rice

from scipy.stats import poisson
from scipy.stats import chi2


def rice2_gen(K, size=1):
    mu2 = K/(K+1)
    sigma2 = 1/(2*(1+K))
    P = poisson.rvs(K, size=size)
    X = chi2.rvs(2*P + 2, size=size)
    return sigma2 * X

class UAV:

    def __init__(self, starting_x, starting_y, starting_z, starting_energy=10,
                 slot_time=1, P0=1, Pi=1, v0=1, lambda3=1):
        self.x = starting_x
        self.y = starting_y
        self.z = starting_z
        self.speed = 0
        # self.z = H
        self.energy = starting_energy  # can be assumed it is +infinity
        self.P0 = (0.012/8) * 1.225 * 0.05 * 0.79 * pow(400, 3) * pow(0.5, 3)
        self.Pi = (1 + 0.1) * pow(100, 1.5) / np.sqrt(2 * 1.225 * 0.79)
        self.lambda1 = pow(200, 2)
        self.lambda2 = 2 * pow(7.2, 2)
        self.lambda3 = 0.5 * 0.3 * 1.225 * 0.05 * 0.79
        self.slot_time = slot_time

    def power_consumption(self):  # need modification

        return self.P0 * (1 + 3 * pow(self.speed, 2) / self.lambda1) +\
               self.Pi * np.sqrt(np.sqrt(1 + pow(self.speed, 4)/pow(self.lambda2, 2)) - pow(self.speed, 2)/self.lambda2)+ \
               self.lambda3 * pow(self.speed, 3)
        #return 0.1 * pow(self.speed, 2)

    def get_internal_state(self, x_lim, y_lim):     ###########
        #return np.array([self.x/x_lim, self.y/y_lim, self.speed/self.max_speed, self.orientation/(2 * np.pi)])
        return np.array([self.speed/self.max_speed, self.orientation/(2 * np.pi)])

    def get_location(self):
        return np.array([self.x, self.y, self.z])

    def comm_radius(self):
        return self.z


class Sensor:

    def __init__(self, x, y, data, starting_energy=1, slot_time=1, t_power=-40):
        self.x = x
        self.y = y
        self.z = 0
        self.energy = starting_energy
        self.slot_time = slot_time
        self.cur_data = data
        self.total_data = data
        self.power = t_power

    def get_location(self):
        return np.array([self.x, self.y, self.z])

    def update_power(self, delta_p):
        self.power = np.clip(delta_p, 0, 0.1)

    def get_power(self):
        return self.power

    def get_power_db(self):
        return pow(10, self.power/10)

    def get_location(self):
        return np.array([self.x, self.y, self.z])

    def send_data(self, rate):
        data_collected = min(rate * self.slot_time, self.cur_data)
        self.cur_data = max(0, self.cur_data - self.slot_time * rate)
        if self.cur_data == 0:
            return True, data_collected
        else:
            return False, data_collected

    def remained(self):
        return self.cur_data/self.total_data


class Dynamics:

    def __init__(self, fixed_p=None, uav_height=10, r_max=1., n_uav=1, n_sensor=1, episode_time_limit=100, data=200):
        self.R_MAX = r_max

        self.data = data
        self.fixed_p = fixed_p
        self.sensor_centers = None
        self.channel_gains = None
        self.Kfactor = None

        self.uav_centers = None
        self.uav_height = uav_height
        self.UAV_blob = None
        self.los = None

        self.state = None
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        self.time_limit = episode_time_limit
        self.time_counter = 0

    def reset(self, Kfactor, locations=None):

        self.Kfactor = Kfactor
        self.locate_objects(locations)
        self.current_target = 0
        # self.in_radius = 1
        h = self.UAV_blob.z
        self.distance = np.linalg.norm(self.uav_centers - self.sensor_centers[0])
        self.d = np.sqrt(pow(self.distance, 2) + pow(h, 2))
        self.ep_step = 0
        # define state
        self.state = self.build_state()
        self.time_counter = 0
        self.energy_consumption = self.UAV_blob.power_consumption()

        return self.state
        # self.UAV_list = [UAV(center[0],center[1]) for center in self.uav_centers]

    def locate_objects(self, locations):
        # self.uav_centers = np.array([0.5, 0.5]) * self.unit_size  # UAV's Starting Point
        self.uav_centers = locations[0]
        self.sensor_centers = locations[1]
        # self.uav_centers = np.array([[0.5,0.5]])    ##UAV's Starting Point
        self.UAV_blob = UAV(self.uav_centers[0][0], self.uav_centers[0][1], self.uav_height)  # MAX Speed = 15
        self.Sensor_blobs = [Sensor(x=centers[0], y=centers[1], data=self.data) for centers in self.sensor_centers]
        # print("ENV-Locate Objects OBS SENSOR", self.obs_centers, self.sensor_centers)

    def build_state(self):             # Normalized State

        # a = (self.sensor_centers - self.uav_centers) / (self.unit_size * np.array([self.Mx, self.My]))
        h = self.UAV_blob.z
        self.distance = np.linalg.norm(self.uav_centers - self.sensor_centers[0])
        d = np.sqrt(pow(self.distance, 2) + pow(h, 2))
        #####################################################################
        self.determine_randomness(h, d)         # self.channel_gain,  self.los
        data_remained = self.Sensor_blobs[0].remained()     # remaining data
        time_remained = 1 - self.ep_step/self.time_limit    # remaining time
        #####################################################################
        # if self.channel_gain < 0.01:
        #     CG_state = 0
        # elif 0.01 < self.channel_gain < 0.1:
        #     CG_state = 0.33
        # elif 0.1 < self.channel_gain < 1:
        #     CG_state = 0.66
        # else:
        #     CG_state = 1
        ######################################################################
        inter = np.array([data_remained, time_remained, self.channel_gain])
        self.raw_state = inter


        return np.concatenate((data_remained, self.channel_gain), axis=None)  # //np.array([0.1, 0.1]

    def step(self, action, fixed_rate=False):
        
        #action = action.numpy()
        # if self.fixed_p:
        #     delta_power = self.fixed_p
        # else:
        #     delta_power = action
        delta_power = action

        if fixed_rate:
            rateph = action
            SNRdb, pw = self.find_power(rateph, self.UAV_blob.z, self.d )
            delta_power = pw
        print("RAW CURRENT STATE", self.raw_state)
        print("CHANNEL GAIN", self.channel_gain)
        print("ACTION Real Power", delta_power)

        # Move the UAV and watch for 1-collision with obstacles, 2-out the border
        # print("radius", "theta", radius, theta)

        sensor_finish = False
        data_reward = 0
        finish_reward = 0
        finish_pen = 0
        self.time_counter += 1      # Starting from 1


        self.Sensor_blobs[0].update_power(delta_power)      # Power which is between Min and Max

        sensor_panalty = self.Sensor_blobs[0].get_power()   # The power is between MIN and MAX
        SNRdb, rateph = self.find_rate(self.Sensor_blobs[0].get_power(), self.UAV_blob.z, self.d)

        sensor_finish, data_collected = self.Sensor_blobs[0].send_data(rateph)
        # update sensor's remaining data, return collected data
        print("SNR db", SNRdb, "RATE",  rateph, "POWER", self.Sensor_blobs[0].get_power())
        # if self.time_counter >= self.time_limit:
        #     if self.Sensor_blobs[0].remained():
        #         finish_pen = 100


        reward = - 0.1 - 10 * sensor_panalty + 0.1 * 1e-6 * data_collected
        #reward = - 0.1 - 10 * sensor_panalty + 0.1 * 1e-6 * data_collected
        # reward = - 0.5 - 10 * sensor_panalty + 0.2 * 1e-6 * data_collected
        # - 0.002 * self.energy_consumption
        print("TOTAL REWARD", reward)
        print("Sensor Penalty", - 10 * sensor_panalty)
        print("DATA Reward", 0.1 * 1e-6 * data_collected)

        self.ep_step += 1
        new_state = self.build_state()
        print("RAW NEW STATE", self.raw_state)
        print("NEW STATE ", new_state)
        return new_state, reward, sensor_finish, sensor_panalty, self.UAV_blob.power_consumption(), rateph, data_collected

    def determine_randomness(self, h, d):
        self.los_prob = self.get_los_prob(h=h, d=d)
        self.los = 1 if random.random() < self.los_prob else 0  # LoS Probability

        if self.channel_gains is None:
        #self.channel_gain = self.los_prob * 1.5 + (1 - self.los_prob * 1.253)
            if self.los:
                #self.channel_gain = (rice.rvs(0.5, size=1) ** 2)[0]
                self.channel_gain = rice2_gen(K=self.Kfactor)[0]
            else:
                self.channel_gain = (np.random.rayleigh(scale=1) ** 2)
        else:
            self.channel_gain = self.channel_gains[self.ep_step]

        if self.channel_gain > 10:
            self.channel_gain = 10

        return self.los, self.channel_gain

    def estimate_channel(self, h, d):
        if self.los:
            pathloss = self.get_pathloss(d=d, los=True)  # dB
        else:
            pathloss = self.get_pathloss(d=d, los=False)

        return pathloss
        #return self.los_prob * self.get_pathloss(d=d, los=True) + (1-self.los_prob) * self.get_pathloss(d=d, los=False)

    def get_los_prob(self, h, d, c1=12.08, c2=0.11):
        return 1 #1/(1+c1*np.exp(-c2 * ((180/np.pi) * np.arcsin(h/d)-c1)))

    def get_pathloss(self, d, los, f=2e9, c=3e8, eta_los=1.6, eta_nlos=23):
        if los:
            return -((20 * np.log10(4 * np.pi * f * d/c) + eta_los))
        else:
            return -((20 * np.log10(4 * np.pi * f * d/c) + eta_nlos))

    def find_rate(self, pt, h, d):
        if pt == 0:
            return -np.Inf, 0
        else:
            pt = 10 * np.log10(pt)
            pathloss = self.estimate_channel(h, d)
            SNRdb = pathloss + 10 * np.log10(self.channel_gain) + pt + 100
            SNR = pow(10, SNRdb/10)
            return SNRdb, 1e6 * np.log2(1 + SNR)

    def find_power(self, rt, h, d):
            #rt = 10**(rt/10)
            #pt = 10 * np.log10(pt)
            pathloss = self.estimate_channel(h, d)
            pt = 10*np.log10(pow(2, rt/1e6) - 1) - 100 - pathloss - 10 * np.log10(self.channel_gain)
            SNRdb = pathloss + 10 * np.log10(self.channel_gain) + pt + 100
            pt = pow(10, pt/10)#10**(pt/10)
            #SNR = pow(10, SNRdb/10)
            return SNRdb, pt



