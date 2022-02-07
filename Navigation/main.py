from NavEnv import UAV, Sensor, Nav_Dynamics
from rlframework import Agent
from rlframework import plot_learning_curve_2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import rice
import time


def plot_trajectory(trajectory, sensors, obstacle, ep_number, ep_step, obs_r, target_r, lim=30):
    # To illustrate the trajectory of the UAV in the environment

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(trajectory[:, 0], trajectory[:, 1], '--bo')
    for sensor in sensors:
        # rect = patches.Rectangle((sensor[0]-obs_radius, sensor[1]-obs_radius), 2*obs_radius, 2*obs_radius,
        #                          linewidth=1, edgecolor='k',  facecolor='g')
        # ax.add_patch(rect)
        ax.scatter(sensor[0], sensor[1], s=100, c='g')
        circle = plt.Circle((sensor[0], sensor[1]), target_r, fill=False, edgecolor='g', linestyle='--')
        ax.add_artist(circle)

    for obs in obstacle:
        rect = patches.Rectangle((obs[0] - obs_r, obs[1] - obs_r), 2*obs_r, 2*obs_r,
                                 linewidth=1, edgecolor='k', facecolor='r')
        ax.add_patch(rect)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"EPISODE {ep_number}, {ep_step} time steps")
    ax.set_xlim(xmin=-0.0, xmax=lim+0)
    ax.set_ylim(ymin=-0.0, ymax=lim+0)


if __name__ == '__main__':

    start_time = time.time()

    print("Main Dynamics is gonna be created")
    n_episodes = 10
    n_sensor = 1

    OBS_RADIUS = 20
    height = 50
    Unit_size = 25
    TARGET_Radius = 20

    SHOW_EVERY = 100
    EPISODE_TIME_LIMIT = 40

    test_mode = True


    score_history = [[]]
    energy_history = [[]]

    for learning_rr, cnt in zip([0.0001], [0]):

        env = Nav_Dynamics(unit_size=Unit_size, uav_height=height, max_x=24, max_y=24, n_uav=1, n_sensor=1,
                           n_obstacle=8, obstacle_r=OBS_RADIUS, target_r=TARGET_Radius, max_rangefinder=100,
                           max_speed=40, time_slot=1, episode_time_limit=EPISODE_TIME_LIMIT, num_rangers=5)

        print("Main Action Space Shape", env.action_space.shape[0])
        agent = Agent(input_dims=env.state_space, env=env, n_actions=env.action_space.shape[0], gamma=1,
                      alpha=learning_rr, beta=10 * learning_rr, batch_size=64, fc1=400, fc2=300, noise=0.2, tau=0.001,
                      start_steps=4000, max_size=50000)

        print("Main Agent is created")

        # env = gym.make('Pendulum-v0')
        # agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0],
        #               alpha=0.001, beta=0.001, batch_size=64, fc1=400, fc2=300, noise=0.2, tau=0.01)

        figure_file = 'plots/uav.png'
        best_score = -float('inf')

        critic_losses = []
        actor_losses = []
        num_steps = 0

        observation = env.reset(schedule=[0], random=True)  # random=TRUE : randomly locate the obstacles and sensor
        agent.initialize_networks(state=observation.reshape(1, -1))  # Create Neural Nets

        # Some predefined locations for the obstacles and sensor when random=FALSE
        obs = [100, 105, 110, 115,
               220, 225, 230, 235,
               340, 345, 350, 355,
               460, 465, 470, 475]
        sens_sch = [25,
                    172, 188,
                    300, 308,
                    365,
                    495]
        uav_ = [50,   55,  60,  65,  70,
                170,  190,
                290,  310,
                410,  430,
                530, 535, 540, 545, 550]
        sens_ = [
                172,   175,  180,  185,  188,
                292, 295, 300, 305, 308,
                412, 415, 420, 425, 428, 495]
        obs_sch = [110,
                   220, 225, 230,
                   345, 355,
                   465, 475]

        uav_sensss = uav_ + sens_
        safe_counter = 0

        if test_mode:
            energy_matrix = np.zeros((n_sensor+1, n_sensor+1))
            agent.load_models(number=None)

            # centers = [0, 85, 95, 195, 185]

            for episode in range(n_episodes):

                # locs = np.array([centers[0]] + obs + centers[1:])
                # observation = env.reset(schedule=[0], random=True)
                # locs = np.array([357, 213, 518, 33, 248, 380, 525])
                # locs = np.random.choice(np.arange(0, 20 * 20), 1 + 10 + 1, replace=False)
                # uav = np.random.choice(uav_, 1, replace=False)
                # uav_sens = np.random.choice(uav_sensss, 2, replace=False)

                sens = np.random.choice(sens_, 2, replace=False)
                print([sens[0]] + obs_sch + [sens[1]])
                locs = np.array([sens[0]] + obs_sch + [sens[1]])
                # locs[0]:starting location index, loc[1:-1]: obstacle indices, loc[-1]: sensor location index
                observation = env.reset(schedule=[0], random=True, locations=locs)
                # random=False : locate the starting point, obstacles and sensor based on "locs"

                done = False
                score, episode_step, energy = 0, 0, 0
                while episode_step < EPISODE_TIME_LIMIT and not done:
                    episode_step += 1
                    num_steps += 1
                    print("####### MAIN EPISODE STEP NUMBER", episode, episode_step)
                    # print("TIME STEP", num_steps)
                    action = agent.choose_action(observation, test_mode)
                    observation_, reward, done, s, trajectory, obs_centers, sensor_centers, energy_i = env.step(action)
                    score += reward
                    energy += energy_i
                    # print("MAIN Action", observation, action, reward, observation_)
                    observation = observation_
                if episode_step != 30:
                    safe_counter += 1
                if episode > (n_episodes-5):
                    plot_trajectory(trajectory, sensor_centers, obs_centers, 1, episode_step, obs_r=OBS_RADIUS,
                                    target_r=TARGET_Radius, lim=24 * Unit_size)
            print(safe_counter)

        else:  # Training

            for episode in range(n_episodes):
                # observation = env.reset(schedule=[0], random=True)
                # locs = np.array([357, 213, 518, 33, 248, 380, 525])
                # uav = np.random.choice(uav_, 1, replace=False)
                # sens = np.random.choice(sens_, 1, replace=False)
                # locs = np.array([uav[0]] + obs + [sens[0]])
                # obsss = list(np.random.choice(obs, 8, replace=False))
                # uav_sens = np.random.choice(sens_sch, 2, replace=False)
                # locs = np.array([uav_sens[0]] + obsss + [uav_sens[1]])
                #locs = np.random.choice(np.arange(0, 25 * 25), 1 + 5 + 1, replace=False)
                observation = env.reset(schedule=[0], random=True)
                done, d_store = False, False
                score, episode_step, energy = 0, 0, 0
                render = (episode % SHOW_EVERY == 0)

                while episode_step < EPISODE_TIME_LIMIT and not done:
                    episode_step += 1
                    num_steps += 1
                    print("####### MAIN EPISODE STEP NUMBER", episode, episode_step)
                    # print("TIME STEP", num_steps)
                    action = agent.choose_action(observation, test_mode)
                    observation_, reward, done, s_reached, trajectory, obs_centers, sensor_centers, energy_i = \
                        env.step(action)
                    score += reward
                    energy += energy_i
                    #d_store = False if episode_step == EPISODE_TIME_LIMIT else done
                    # print("MAIN Action", observation, action, reward, observation_)
                    agent.remember(observation, action, reward, observation_, done)
                    agent.learn()
                    observation = observation_

                score_history[cnt].append(score)
                energy_history[cnt].append(energy)

                if episode > (n_episodes - 5):  # Plot the trajectories for the last 5 episodes
                    plot_trajectory(trajectory, sensor_centers, obs_centers, episode, episode_step, obs_r=OBS_RADIUS,
                                    target_r=TARGET_Radius, lim=24 * Unit_size)

                # env.trans_reward_decay *= 0.999
                print("TRANS DECAYY", env.trans_reward_decay)
                avg_score = np.mean(score_history[cnt][-SHOW_EVERY:])
                if not test_mode and avg_score > best_score:
                    best_score = avg_score
                    agent.save_models()
                if episode % 100 == 0:
                    agent.dev *= 0.99


            print('episode ', episode, "Time Step", episode_step, 'score %.1f' % score, 'avg score %.1f' % avg_score)
            print("#############################")

        print(score_history[cnt])

    plot_learning_curve_2(score_history, SHOW_EVERY, "Number of episodes used in training", "Average Return",
                        [0, n_episodes], figure_file)
    plot_learning_curve_2(energy_history, SHOW_EVERY, "Number of episodes used in training","Average Propulsion Energy",
                        [0, n_episodes], figure_file)
    plt.show()
    plt.close('all')

    print("--- %s seconds ---" % (time.time() - start_time))