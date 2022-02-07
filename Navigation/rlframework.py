import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam
import datetime
import random



class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):  # Number of action components
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
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
                 name='actor', chkpt_dir='models/'):
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


class Agent:
    def __init__(self, alpha=0.001, beta=0.001, input_dims=[8], env=None,
                 gamma=0.99, n_actions=2, max_size=100000, tau=0.001,
                 fc1=400, fc2=300, batch_size=64, noise=0.2, start_steps=1000):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        #self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        # self.noise = OUActionNoise(mu=np.zeros(1), sigma=float(noise) * np.ones(1))
        self.dev = noise
        self.start_steps = start_steps

        self.actor = ActorNetwork(n_actions=n_actions, name='actor',
                                  fc1_dims=fc1, fc2_dims=fc2, max_action=self.max_action)
        self.critic = CriticNetwork(n_actions=n_actions, name='critic',
                                    fc1_dims=fc1, fc2_dims=fc2)
        self.target_actor = ActorNetwork(n_actions=n_actions,
                                         name='target_actor', fc1_dims=fc1, fc2_dims=fc2, max_action=self.max_action)
        self.target_critic = CriticNetwork(n_actions=n_actions,  name='target_critic',
                                           fc1_dims=fc1, fc2_dims=fc2)

        # self.actor = self.Actor_gen(state_size=self.num_states, action_size=n_actions, name='actor',
        #                             hidden_layers=[fc1, fc2], max_action=self.max_action)
        # self.critic = self.Critic_gen(state_size=self.num_states, action_size=n_actions, name='critic',
        #                               hidden_layers=[fc1, fc2])
        # self.target_actor = self.Actor_gen(state_size=self.num_states, action_size=n_actions, name='target_actor',
        #                                    hidden_layers=[fc1, fc2], max_action=self.max_action)
        # self.target_critic = self.Critic_gen(state_size=self.num_states, action_size=n_actions, name='target_critic',
        #                                      hidden_layers=[fc1, fc2])

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

    # def Critic_gen(self, state_size, action_size, hidden_layers, name):
    #     input_x = Input(shape=state_size)
    #     input_a = Input(shape=action_size)
    #     out = concatenate([input_x, input_a], axis=-1)
    #     for i in hidden_layers:
    #         out = Dense(i, activation='relu')(out)
    #     out = Dense(1, activation=None)(out)
    #     critic_model = tf.keras.Model(inputs=[input_x, input_a], outputs=out, name=name)
    #     return critic_model
    #
    # def Actor_gen(self, state_size, action_size, hidden_layers, name, max_action=1):
    #     input_x = Input(shape=state_size)
    #     out = Dense(hidden_layers[0], activation='relu')(input_x)
    #     for i in hidden_layers[1:]:
    #         out = Dense(i, activation='relu')(out)
    #     out = Dense(action_size, activation='tanh')(out)
    #     # x = tf.math.multiply(x, action_mult)
    #     act_model = tf.keras.Model(inputs=input_x, outputs=out, name=name)
    #     return act_model

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

        # temp1 = np.array(self.target_critic.get_weights())
        # temp2 = np.array(self.critic.get_weights())
        # temp3 = tau * temp2 + (1 - tau) * temp1
        # self.target_critic.set_weights(temp3)
        #
        # # updating Actor network
        # temp1 = np.array(self.target_actor.get_weights())
        # temp2 = np.array(self.actor.get_weights())
        # temp3 = tau * temp2 + (1 - tau) * temp1
        # self.target_actor.set_weights(temp3)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self, number=None):
        print('... saving models ...')
        if number is None:
            self.actor.save_weights(self.actor.checkpoint_file + '.h5')
            self.target_actor.save_weights(self.target_actor.checkpoint_file + '.h5')
            self.critic.save_weights(self.critic.checkpoint_file + '.h5')
            self.target_critic.save_weights(self.target_critic.checkpoint_file + '.h5')
        else:
            self.actor.save_weights(self.actor.checkpoint_file + str(number) + '.h5')
            self.target_actor.save_weights(self.target_actor.checkpoint_file + str(number) + '.h5')
            self.critic.save_weights(self.critic.checkpoint_file + str(number) + '.h5')
            self.target_critic.save_weights(self.target_critic.checkpoint_file + str(number) + '.h5')

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

    def choose_action(self, observation, evaluate=False):
        # print("Agent Choose Action...")
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        if not evaluate:
            if self.memory.mem_cntr > self.start_steps:
                actions = self.actor(state) # * self.max_action
                noise = tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.dev)
                # noise = self.noise()
                # print("NOISE", noise)
                actions += noise
                # actions += tf.random.uniform(shape=[self.n_actions], minval=-0.25, maxval=0.25)
                actions = tf.clip_by_value(actions, self.min_action, self.max_action)
                print("Action", actions, "Noise", noise)
            else:
                actions = tf.random.uniform(shape=[self.n_actions, 1], minval=self.min_action, maxval=self.max_action)
        else:
            actions = self.actor(state) # * self.max_action

        return actions[0]

    def learn(self):
        if self.memory.mem_cntr <= self.start_steps:
            return None, None

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        #  print("LEARN SHAPE ", state.shape, action.shape, reward.shape, new_state.shape )

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states) # * self.max_action
            Q = self.critic(states, new_policy_actions)
            actor_loss = -tf.reduce_mean(Q)
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        with tf.GradientTape() as tape2:
            target_actions = self.target_actor(states_) # * self.max_action
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma * critic_value_ * (1 - done)
            # critic_loss = keras.losses.MSE(target, critic_value)
            critic_loss = tf.reduce_mean((critic_value - target) ** 2)
            reg_loss = tf.reduce_sum(self.critic.losses)
            total_loss = critic_loss + reg_loss
        critic_network_gradient = tape2.gradient(total_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))
        #print(done)
        self.update_network_parameters()
        return critic_loss, actor_loss


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


