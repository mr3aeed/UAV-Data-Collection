import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import datetime
from gym import wrappers


class ReplayBuffer:
    def __init__(self, max_size, input_shape):  # Number of action components
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        # print("input shape", input_shape)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size=64):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        # print(batch)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones


class CriticNetwork(keras.Model):

    def __init__(self, n_actions, fc1_dims=400, fc2_dims=300, name='critic', chkpt_dir='models/'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        # self.q = Dense(1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01),
        #                kernel_initializer=last_init)
        self.q = Dense(self.n_actions, activation=None, kernel_initializer=last_init, bias_initializer=last_init)

    def call(self, state):
        action_value = self.fc1(state)
        action_value = self.fc2(action_value)
        q = self.q(action_value)
        return q


class DQN:
    def __init__(self, name, input_dims, num_actions, gamma=0.99, max_size=1000000, start_steps=500,
                 epsilon=0.99, min_epsilon=0.1, batch_size=64, lr=0.001, fc1=256, fc2=256):
        self.action = [i for i in range(num_actions)]
        self.num_actions = num_actions
        self.start_steps = start_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.name = name
        
        self.memory = ReplayBuffer(max_size, input_dims)
        self.critic = CriticNetwork(n_actions=num_actions, name='critic', fc1_dims=fc1, fc2_dims=fc2)
        self.target_critic = CriticNetwork(n_actions=num_actions, name='target_critic', fc1_dims=fc1, fc2_dims=fc2)
        self.critic.compile(optimizer=Adam(learning_rate=lr))
        self.target_critic.compile(optimizer=Adam(learning_rate=lr))

    def initialize_networks(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        dummy_Q = self.critic(state)
        dummy_Q = self.target_critic(state)
        self.update_network_parameters()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation, evaluate):
        # print("Agent Choose Action...")
        if not evaluate:
            if np.random.random() < self.epsilon:
                print("RANDOM")
                action = np.random.choice(self.action)
            else:
                print("NETWORK")
                state = tf.convert_to_tensor([observation], dtype=tf.float32)
                actions_p = self.critic(state)
                #print(actions_p)
                action = tf.math.argmax(actions_p, axis=1)[0].numpy()
        else:
            print("NETWORK")
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            actions_p = self.critic(state)
            # print(actions_p)
            action = tf.math.argmax(actions_p, axis=1)[0].numpy()
            
        return action

    # def get_action(self, states, epsilon):
    #     if np.random.random() < epsilon:
    #         return np.random.choice(self.num_actions)
    #     else:
    #         return np.argmax(self.predict(np.atleast_2d(states))[0])
    #

    def learn(self):
        if self.memory.mem_cntr <= self.start_steps:
            return None
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        #print("LEARN SHAPE ", state.shape, action.shape, reward.shape, new_state.shape )
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.int32)

        with tf.GradientTape() as tape:
            critic_value_ = np.max(self.target_critic(states_), axis=1)
            target = rewards + self.gamma * critic_value_ * (1 - done)
            critic_value = tf.math.reduce_sum(self.critic(states) * tf.one_hot(action, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(target - critic_value))
            # reg_loss = tf.reduce_sum(self.critic.losses)
            # total_loss = loss + reg_loss
        critic_network_gradient = tape.gradient(loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))
        return loss

    def update_network_parameters(self):
        temp = np.array(self.critic.get_weights())
        self.target_critic.set_weights(temp)

    # def add_experience(self, exp):
    #     if len(self.experience['s']) >= self.max_experiences:
    #         for key in self.experience.keys():
    #             self.experience[key].pop(0)
    #     for key, value in exp.items():
    #         self.experience[key].append(value)
    #
    # def copy_weights(self, TrainNet):
    #     variables1 = self.model.trainable_variables
    #     variables2 = TrainNet.model.trainable_variables
    #     for v1, v2 in zip(variables1, variables2):
    #         v1.assign(v2.numpy())
    def load_models(self, number=None):
        print('... loading models ...')
        if number is None:        
            self.critic.load_weights(self.critic.checkpoint_file + self.name + '.h5')
            self.target_critic.load_weights(self.target_critic.checkpoint_file + self.name + '.h5')
    
    def save_models(self, number=None):
        print('... saving models ...')
        if number is None:
            self.critic.save_weights(self.critic.checkpoint_file + self.name + '.h5')
            self.target_critic.save_weights(self.target_critic.checkpoint_file + self.name + '.h5')

# def plot_learning_curve(scores, how_many, title, figure_file):
#     running_avg = np.zeros(len(scores))
#     for i in range(len(running_avg)):
#         running_avg[i] = np.mean(scores[max(0, i - how_many):(i + 1)])
#     plt.figure()
#     # plt.subplot(1, 2, 1)
#     plt.plot(running_avg)
#     plt.title(f"Running average of previous {how_many} " + title)
#     # plt.subplot(1, 2, 2)
#     # plt.plot(x, scores)
#     # plt.title('Scores')
#     plt.show()


def plot_learning_curve(total_score, how_many, xlabel,  ylabel, xlim):

    #total_score = np.sum(scores, axis=0)
    x = np.arange(xlim[0], xlim[1])
    f = interp1d(x, total_score, kind='cubic')

    length = total_score.shape[0]
    running_avg = np.zeros(length)
    for i in range(length):
        running_avg[i] = np.mean(total_score[max(0, i - how_many):(i + 1)])
    # for i in range(len(running_avg)):
    #     running_avg[1, i] = np.mean(scores2[max(0, i - how_many):(i + 1)])

    plt.figure()
    plt.plot(running_avg)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim[0], xlim[1])

    plt.figure()
    # plt.subplot(1, 2, 1)

    xnew = np.linspace(xlim[0], xlim[1]-1, num=int((xlim[1]-xlim[0])/30), endpoint=True)
    plt.plot(x, total_score, 'b', xnew, f(xnew), 'r-')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim[0], xlim[1])

    plt.show()