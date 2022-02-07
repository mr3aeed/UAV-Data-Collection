from DQN import DQN, plot_learning_curve
from SchedulingEnv2 import Scheduler
import gym
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches


if __name__ == '__main__':

    print("Main Dynamics is gonna be created")
    n_episodes = 100
    SHOW_EVERY = 1
    EPISODE_TIME_LIMIT = 100
    test_mode = False

    n_sensor = 6
    n_obstacle = 8
    n_uav = 2
    num_steps = 0
    score_history = np.zeros((n_uav, n_episodes))
    time_history = np.zeros(n_episodes)
    best_scores = [-float('inf'), -float('inf')]
    best_score = -float('inf')

    env = Scheduler(unit_size=25, uav_height=25, obstacle_r=20, max_x=24, max_y=24, n_uav=n_uav, n_sensor=n_sensor,
                    n_obstacle=n_obstacle, max_rangefinder=100, num_rangers=5, episode_time_limit=20, target_r=20,
                    max_speed=40, nav_time_slot=1, com_slot_time=1)
    agents = [DQN(name=name, input_dims=[2*n_uav + n_sensor], num_actions=n_sensor+1, lr=0.001, batch_size=64,
                  epsilon=0.25, min_epsilon=0.05, fc1=300, fc2=200) for name in ('a1', 'a2')]

    obs_sensor_list = np.array([110, 220, 225, 230, 345, 355, 465, 475,
                                172,  188, 300, 308, 365, 495])
    observation = env.reset(random=False, locations=obs_sensor_list)
    for i in range(n_uav):
        agents[i].initialize_networks(state=observation.reshape(1, -1))

    if test_mode:
        for i in range(n_uav):
            agents[i].load_models()
        for episode in range(n_episodes):
            done = False
            score0, score1, episode_step = 0, 0, 0
            observation = env.reset(random=False, locations=obs_sensor_list)

            while not done:

                episode_step += 1
                num_steps += 1
                print("####### MAIN EPISODE STEP NUMBER", episode, episode_step)
                actions = [agents[i].choose_action(observation, test_mode) for i in range(n_uav)]
                #actions = actions_set[episode_step-1]
                observation_, rewards, done = env.step(actions)
                score0 += rewards[0]
                score1 += rewards[1]
                observation = observation_
                if episode_step >= EPISODE_TIME_LIMIT:
                    done = True
            env.get_trajectories(episode)
            score_history[0, episode] = score0
            score_history[1, episode] = score1
            #if episode > n_episodes - 2:

    else:
        for episode in range(n_episodes):
            done = False
            score0, score1, episode_step = 0, 0, 0
            observation = env.reset(random=False, locations=obs_sensor_list)

            while not done:
                episode_step += 1
                num_steps += 1
                print("####### MAIN EPISODE STEP NUMBER", episode, episode_step)
                actions = [agents[i].choose_action(observation, test_mode) for i in range(n_uav)]
                # actions = [0, 1]
                observation_, rewards, done = env.step(actions)
                score0 += rewards[0]
                score1 += rewards[1]
                d_store = False if episode_step == EPISODE_TIME_LIMIT else done
                for i in range(n_uav):
                    agents[i].remember(observation, actions[i], rewards[i], observation_, done)
                    #agents[i].remember(observation, actions[i], sum(rewards), observation_, done)
                    # Here we can store the overall reward or the individual reward
                    c = agents[i].learn()
                    if num_steps % 50 == 0:
                        agents[i].update_network_parameters()
                observation = observation_
                if episode_step >= EPISODE_TIME_LIMIT:
                    done = True

            for i in range(2):
                agents[i].epsilon = max(agents[i].min_epsilon, agents[i].epsilon * 0.998)
            if episode > n_episodes - 2:
                env.get_trajectories(episode)

            score_history[0, episode] = score0
            score_history[1, episode] = score1
            time_history[episode] = episode_step
            scores = [score0, score1]

            for i in range(n_uav):
                if scores[i] >= best_scores[i]:
                    best_scores[i] = scores[i]
                    agents[i].save_models()

        plot_learning_curve(np.sum(score_history, axis=0), SHOW_EVERY, "Episode number", "Return", [0, n_episodes])
        plot_learning_curve(time_history, SHOW_EVERY, "Episode number", "Mission Time", [0, n_episodes])
        # plt.figure()
        # plt.plot(time_history)
        # plt.grid()
        # plt.xlabel('Episode number')
        # plt.ylabel('Mission Time')
        # plt.xlim(0 , n_episodes)

    plt.show()
