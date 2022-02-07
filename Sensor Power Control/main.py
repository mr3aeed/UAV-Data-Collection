from comm_env import UAV, Sensor, Dynamics, rice2_gen
from rlframework import Agent
import matplotlib.pyplot as plt
from rlframework import plot_learning_curve
import numpy as np
from scipy.stats import rice
import os.path


def determine_randomness(episode_n, length, Kfactor):  # Generate the channel gains according to Rician Distribution
    #if not os.path.isfile('channel.npy'):
    print('The Gains Matrix is being generated')
    array = np.zeros((episode_n, length))
    for j in range(episode_n):
        for i in range(length):
            channel_gain = rice2_gen(Kfactor, size=1)[0]
            array[j, i] = channel_gain
    np.save('channel.npy', array)
    # else:
    #     array = np.load('channel.npy')
    return array


def determine_randomness_2(episode_n, length, Goodfirst=True):  # Simple Scenario: switch between 0 and 1
    array = np.zeros((episode_n, length))
    for j in range(episode_n):
        channel = False
        for i in range(length):
            if i % 10 == 0:
                channel = not channel
            if channel:
                array[j, i] = 1
            else:
                array[j, i] = 0.1

    return array


def determine_average_power(array=[], load=True):
    if load:
        array = np.load('avg_power.npy')
    else:
        np.save('avg_power.npy', array)
    return array


def determine_average_rate(array=[], load=True):
    if load:
        array = np.load('avg_rate.npy')
    else:
        np.save('avg_rate.npy', array)
    return array


if __name__ == '__main__':

    print("Main Dynamics is gonna be created")
    n_episodes = 20
    SHOW_EVERY = 100
    EPISODE_TIME_LIMIT = 100
    DATA = 100e6
    height = 50
    k_factor = 1
    test_mode = True

    k_factors = [0.1, 1, 10]
    gains_matrix = determine_randomness(1000, 100, Kfactor=k_factor)
    # gains_matrix = determine_randomness_2(1000, 100, Goodfirst=True)

    env = Dynamics(uav_height=height, episode_time_limit=EPISODE_TIME_LIMIT, data=DATA, fixed_p=None)
    agent = Agent(input_dims=[2], env=env, n_actions=env.action_space.shape[0],
                  alpha=0.001, beta=0.001, batch_size=64, fc1=256, fc2=128, noise=0.2, tau=0.001)

    uav = [[0, 0]]
    sensor = [[0, 0]]
    observation = env.reset(Kfactor=k_factor, locations=np.array([uav, sensor]))
    # observation = env.reset()
    best_score = -float('inf')
    ## agent_dqn.initialize_networks(state=observation.reshape(1, -1))
    agent.initialize_networks(state=observation.reshape(1, -1))

    num_steps = 0

    score_history = []
    energy_history = []
    hover_history = []
    time_steps = []
    rate_history = []

    score_history_test = []
    energy_history_test = []
    hover_history_test = []
    time_steps_test = []
    rate_history_test = []

    average_power = []
    average_rate = []

    score_history2_test = []
    energy_history2_test = []
    energy_history_all_fixed_test = []
    hover_history_all_fixed_test = []
    time_steps2_test = []

    score_history3_test = []
    energy_history3_test = []
    energy_history_all_fixedrate_test = []
    hover_history_all_fixedrate_test = []
    time_steps3_test = []

    if test_mode:

        # fixed_powers = [0.01, 0.02, 0.03, 0.05]
        agent.load_models(number=None)
        time_step_power = []
        time_step_channel = []
        time_step_rate = []

        for episode in range(n_episodes):  # Getting the results of our DRL algorithm
            env.channel_gains = gains_matrix[episode]
            observation = env.reset(Kfactor=k_factor, locations=np.array([uav, sensor]))
            done = False
            score, episode_step = 0, 0
            energy, hover, rate, data = 0, 0, 0, 0

            while episode_step < EPISODE_TIME_LIMIT and not done:
                episode_step += 1
                print("####### MAIN EPISODE STEP NUMBER", episode, episode_step)
                # print("TIME STEP", num_steps)
                print("Observation", observation)
                action = agent.choose_action(observation, test_mode)
                action2 = action * 0.05 + 0.05
                energy_before = energy
                data_before = data

                observation_, reward, done, energy_i, hover_i, rate_i, data_i = \
                    env.step(action2)

                score += reward
                energy += energy_i
                hover += hover_i
                rate += rate_i
                data += data_i

                observation = observation_

                if episode == 0:
                    time_step_power.append(energy_i)
                    time_step_channel.append(env.channel_gains[episode_step]* 0.1 )



            score_history_test.append(score)
            hover_history_test.append(hover)
            energy_history_test.append(energy)
            time_steps_test.append(episode_step)
            rate_history_test.append(rate)

            average_power.append(energy / episode_step)
            average_rate.append(rate / episode_step)

################################# Visualization Part #######################################
        plt.figure()
        #plt.plot( time_step_channel, 'bs--')
        plt.plot( time_step_power, 'bo--')
        #plt.plot(energy_history2_test, 'ro--', label="Fixed Power (Average)")
        #  plt.ylim(30, 55)
        # plt.plot(energy_history3, label="Fixed Rate")
        plt.ylabel("Transmit Power")
        plt.xlabel("Time step")
        # plt.xlim(0, n_episodes + 1)
        #plt.xticks([1, 5, 10, 15, 20])
        # plt.ylim(0.4, 0.55)
        plt.grid()
        #plt.legend(loc="upper right")

        print(env.channel_gains)
        determine_average_power(average_power, load=False)
        determine_average_rate(average_rate, load=False)
        ####################### Fixed Power ###############################
        fixed_powers = determine_average_power(load=True)
        fixed_rates = determine_average_rate(load=True)
        fixed_powers = average_power
        fixed_rates = average_rate

        # for fixed_power in fixed_powers:    # Getting the results of the fixed power approaches
        energy_history2_test = []
        hover_history2_test = []
        for episode in range(n_episodes):

            env.channel_gains = gains_matrix[episode]
            observation = env.reset(Kfactor=k_factor, locations=np.array([uav, sensor]))
            done = False
            score2, episode_step2 = 0, 0
            energy2, hover2, rate2 = 0, 0, 0

            while episode_step2 < EPISODE_TIME_LIMIT and not done:
                print("####### MAIN EPISODE STEP NUMBER", episode, episode_step2)
                episode_step2 += 1
                observation_, reward, done, energy_i, hover_i, rate_i, data_i = env.step(fixed_powers[episode])

                score2 += reward
                energy2 += energy_i
                hover2 += hover_i
                rate2 += rate_i

                observation = observation_

            score_history2_test.append(score2)
            energy_history2_test.append(energy2)
            hover_history2_test.append(hover2)
            rate_history_test.append(rate2)
            time_steps2_test.append(episode_step2)

        print(energy_history_test)
        print(energy_history2_test)
        print(time_steps_test, time_steps2_test)

        ############################ Fixed Rate ############################

        # for episode in range(n_episodes):
        #
        #     env.channel_gains = gains_matrix[episode]
        #     observation = env.reset(Kfactor=k_factor, locations=np.array([uav, sensor]))
        #     done = False
        #     score3, episode_step3 = 0, 0
        #     energy3, hover3, rate3 = 0, 0, 0
        #
        #     while episode_step3 < EPISODE_TIME_LIMIT and not done:
        #         print("####### MAIN EPISODE STEP NUMBER", episode, episode_step3)
        #         episode_step3 += 1
        #         observation_, reward, done, energy_i, hover_i, rate_i = env.step(fixed_rates[episode],
        #                                                                          fixed_rate=True)
        #
        #         score3 += reward
        #         energy3 += energy_i
        #         hover3 += hover_i
        #         rate3 += rate_i
        #
        #         observation = observation_
        #
        #     score_history3_test.append(score3)
        #     energy_history3_test.append(energy3)
        #     #hover_history3_test.append(hover3)
        #     #rate_history_test.append(rate3)
        #     time_steps3_test.append(episode_step3)

###########################################################################
        # plt.figure()
        # plt.plot(np.arange(1, n_episodes+1), energy_history_test, 'bs--', label='DDPG')
        # plt.plot(np.arange(1, n_episodes+1), energy_history2_test, 'ro--', label="Fixed Power (Average)")
        # #  plt.ylim(30, 55)
        # # plt.plot(energy_history3, label="Fixed Rate")
        # plt.ylabel("Sensor Energy Consumption")
        # plt.xlabel("Test Episodes")
        # plt.xlim(0, n_episodes+1)
        # plt.xticks([1, 5, 10, 15, 20])
        # plt.ylim(0.38, 0.5)
        # plt.grid()
        # plt.legend(loc="upper right")
############################################################################
        plt.figure()
        plt.plot(np.arange(1, n_episodes+1), time_steps_test, 'bs--', label='DDPG')
        plt.plot(np.arange(1, n_episodes+1), time_steps2_test, 'ro--', label="Fixed Power (Average)")
        #  plt.ylim(30, 55)
        # plt.plot(energy_history3, label="Fixed Rate")
        plt.ylabel("Completion Time")
        plt.xlabel("Test Episodes")
        plt.xlim(0, n_episodes+1)
        plt.ylim(40, 80)
        plt.xticks([1, 5, 10, 15, 20])
        plt.grid()
        plt.legend(loc="upper right")
##########################################################################
        # plt.figure()
        # plt.plot(np.arange(1, n_episodes + 1), energy_history_test, 'bs--', label='DDPG')
        # plt.plot(np.arange(1, n_episodes + 1), energy_history3_test, 'yx--', label="Fixed Rate (Average)")
        #
        # plt.ylabel("Sensor Energy Consumption")
        # plt.xlabel("Test Episodes")
        # plt.xlim(0, n_episodes + 1)
        # plt.xticks([1, 5, 10, 15, 20])
        # plt.ylim(0.3, 1.5)
        # plt.grid()
        # plt.legend(loc="upper right")
############################################################################
        # plt.figure()
        # plt.plot(np.arange(1, n_episodes + 1), time_steps_test, 'bs--', label='DDPG')
        # plt.plot(np.arange(1, n_episodes + 1), time_steps3_test, 'yx--', label="Fixed Rate (Average)")
        # #  plt.ylim(30, 55)
        # # plt.plot(energy_history3, label="Fixed Rate")
        # plt.ylabel("Completion Time")
        # plt.xlabel("Test Episodes")
        # plt.xlim(0, n_episodes + 1)
        # plt.ylim(40, 80)
        # plt.xticks([1, 5, 10, 15, 20])
        # plt.grid()
        # plt.legend(loc="upper right")

        # energy_history_all_fixed.append(energy_history2)
        # hover_history_all_fixed.append(hover_history2)
        #
        # arr3 = []
        # for i in range(len(fixed_powers)):
        #     x = np.array(energy_history).flatten() < np.array(energy_history2[i])
        #     arr3.append(np.sum(x) / len(x))
        #
        # print(arr3)
        # plt.figure()
        # plt.plot(energy_history, 'r', label='DDPG')
        # for i in range (len(fixed_powers)):
        #     plt.plot(energy_history_all_fixed[i],  label=f"Fixed Power = {fixed_powers[i]}")
        # #plt.ylim(30, 55)
        # plt.ylabel("Sensor Penalty")
        # plt.xlabel("Test Episodes")
        # plt.xlim(0, n_episodes)
        # plt.legend(loc="upper left")
        #
        # plt.figure()
        # plt.plot(hover_history, 'r', label='DDPG')
        # for i in range(len(fixed_powers)):
        #     plt.plot(hover_history_all_fixed[i], label=f"Fixed Power = {fixed_powers[i]}")
        # # plt.ylim(30, 55)
        # plt.ylabel("UAV Power Consumption")
        # plt.xlabel("Test Episodes")
        # plt.xlim(0, n_episodes)
        # plt.legend(loc="upper left")

    ##############################   Training    ################################
    else:

        for episode in range(n_episodes):

            observation = env.reset(Kfactor=k_factor, locations=np.array([uav, sensor]))
            done, d_store = False, False
            score, episode_step = 0, 0
            energy, hover, rate, data = 0, 0, 0, 0
            render = (episode % SHOW_EVERY == 0)

            while episode_step < EPISODE_TIME_LIMIT and not done:
                episode_step += 1
                num_steps += 1
                print("####### MAIN EPISODE STEP NUMBER #########", episode, episode_step)
                # print("TIME STEP", num_steps)
                action = agent.choose_action(observation, test_mode)
                action2 = action * 0.05 + 0.05
                # action_dis = agent_dqn.choose_action(observation, test_mode)
                # action = 0.01 * action_dis
                # observation_, reward, done, _ = env.step(action_dis)
                energy_before = energy
                data_before = rate

                observation_, reward, done, energy_i, hover_i, rate_i, data_i = \
                    env.step(action2)
                score += reward
                energy += energy_i
                hover += hover_i
                rate += rate_i
                data = data_i

                # d_store = False if episode_step == EPISODE_TIME_LIMIT else done
                agent.remember(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_

            score_history.append(score)
            time_steps.append(episode_step)
            energy_history.append(energy)       # Sensor Energy
            rate_history.append(rate)

            avg_score = np.mean(score_history[-SHOW_EVERY:])
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

        plot_learning_curve(score_history, SHOW_EVERY, "Number of episodes",  "Average Return ", [0, n_episodes],
                            figure_file='score.png')
        plot_learning_curve(time_steps, 1, "Number of episodes", "Completion Time", [0, n_episodes],
                            figure_file='completion_time.png')
        plot_learning_curve(energy_history, SHOW_EVERY, "Number of episodes", "Sensor Power Consumption", [0, n_episodes],
                            figure_file='sensor_power.png')
        # plot_learning_curve(hover_history, SHOW_EVERY, "Number of episodes", "UAV Power Consumption", [0, n_episodes],
        #                     figure_file='uav_power.png')
        plot_learning_curve(rate_history, SHOW_EVERY, "Number of episodes", "Data Rate", [0, n_episodes],
                            figure_file='data_rate.png')

    plt.show()