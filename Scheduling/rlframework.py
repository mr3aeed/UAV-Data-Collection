import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam
import datetime
import random


class CriticNetwork(keras.Model):

    def __init__(self, n_actions, fc1_dims=400, fc2_dims=300, name='critic', chkpt_dir='../Navigation/models/'):

        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg')
        self.fc1 = Dense(self.fc1_dims, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu')
        self.fc2 = Dense(self.fc2_dims,  kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu')
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.q = Dense(1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer=last_init)
        #self.q = Dense(1, activation=None, kernel_initializer=last_init, bias_initializer=last_init)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        q = self.q(action_value)
        return q


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=400, fc2_dims=300, n_actions=2, max_action=1,
                 name='actor', chkpt_dir='../Navigation/models/'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.max_action = max_action
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg')
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.mu = Dense(self.n_actions, activation='tanh', kernel_initializer=last_init, bias_initializer=last_init)
        # self.mu = Dense(self.n_actions, activation='tanh', kernel_initializer=last_init)

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob) * self.max_action
        return mu


class Nav_Agent:
    def __init__(self, alpha=0.001, beta=0.001, input_dims=[9], env=None,
                 gamma=0.99, n_actions=2, max_size=100000, tau=0.001,
                 fc1=400, fc2=300, batch_size=64, noise=0.2, start_steps=1000):


        self.num_actions = n_actions
        self.n_actions = n_actions
        self.max_action = 1
        self.min_action = -1

        self.actor = ActorNetwork(n_actions=n_actions, name='actor',
                                  fc1_dims=fc1, fc2_dims=fc2, max_action=self.max_action)
        self.critic = CriticNetwork(n_actions=n_actions, name='critic',
                                    fc1_dims=fc1, fc2_dims=fc2)
        self.target_actor = ActorNetwork(n_actions=n_actions,
                                         name='target_actor', fc1_dims=fc1, fc2_dims=fc2, max_action=self.max_action)
        self.target_critic = CriticNetwork(n_actions=n_actions,  name='target_critic',
                                           fc1_dims=fc1, fc2_dims=fc2)

        lr_schedule_actor = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=alpha, decay_steps=5000, decay_rate=0.95)
        lr_schedule_critic = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=beta, decay_steps=5000, decay_rate=0.95)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        # self.update_network_parameters(tau=1)

    def initialize_networks(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        dummy_action = self.actor(state)
        dummy_Q = self.critic(state, dummy_action)
        dummy_action = self.target_actor(state)
        dummy_Q = self.target_critic(state, dummy_action)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        # print((self.actor.weights[0]).shape, (self.target_actor.weights[0].shape))
        # print(self.critic.weights[0], self.target_critic.weights[0])

    def load_models(self, number=None):
        print('... loading models ...')
        if number is None:
            self.actor.load_weights(self.actor.checkpoint_file + '.h5')
            self.target_actor.load_weights(self.target_actor.checkpoint_file + '.h5')
            self.critic.load_weights(self.critic.checkpoint_file + '.h5')
            self.target_critic.load_weights(self.target_critic.checkpoint_file + '.h5')
        else:
            self.actor.load_weights(self.actor.checkpoint_file + str(number) + '.h5')
            self.target_actor.load_weights(self.target_actor.checkpoint_file + str(number) + '.h5')
            self.critic.load_weights(self.critic.checkpoint_file + str(number) + '.h5')
            self.target_critic.load_weights(self.target_critic.checkpoint_file + str(number) + '.h5')

    def choose_action(self, observation):
        # print("Agent Choose Action...")
        state = tf.convert_to_tensor([observation], dtype=tf.float32)

        actions = self.actor(state)  # * self.max_action

        return actions[0]


def plot_learning_curve(scores, how_many, xlabel,  ylabel, xlim, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - how_many):(i + 1)])
    plt.figure()
    # plt.subplot(1, 2, 1)
    plt.plot(running_avg)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim[0], xlim[1])
    #plt.title(f"Running average of last {how_many} " + title)
    # plt.subplot(1, 2, 2)
    # plt.plot(x, scores)
    # plt.title('Scores')

def plot_learning_curve_2(scores, how_many, xlabel,  ylabel, xlim, figure_file):
    running_avg1 = np.zeros(len(scores[0]))
    #running_avg2 = np.zeros(len(scores[1]))

    for i in range(len(running_avg1)):
        running_avg1[i] = np.mean(scores[0][max(0, i - how_many):(i + 1)])
    # for i in range(len(running_avg2)):
    #     running_avg2[i] = np.mean(scores[1][max(0, i - how_many):(i + 1)])
    plt.figure()
    # plt.subplot(1, 2, 1)
    plt.plot(running_avg1, label="learning rates: 0.0001, 0.001")
    #plt.plot(running_avg2, label="learning rates: 0.001, 0.01")
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim[0], xlim[1])
    #plt.legend(loc="upper right")


