from DeepRM.envs.DeepRMEnv import DeepEnv
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import parameters
import job_distribution
import argparse
import os
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines import DQN
from stable_baselines import TRPO
from stable_baselines import ACKTR
from stable_baselines.common.env_checker import check_env
from matplotlib.cbook import flatten
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common import make_vec_env
from statistics import mean
from collections import defaultdict
import other_agents
from datetime import datetime
import json
import run
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == '__main__':
    # Read the content from the author's implementation
    def read_from_file(pa):
        with open('loadresults.json') as json_file:
            data = json.load(json_file)
            hong_dict = json.loads(data)

            for j in range(len(hong_dict['new_job_rate'])):
                if hong_dict['new_job_rate'][j] == pa.new_job_rate:
                    slowdown = hong_dict['job_slowdown'][j]
                    completion_time = hong_dict['job_completion_time'][j]
                    reward = 0
        return slowdown, completion_time, reward

    # Plot the load variation graph
    def plot_load_variation(load_occupied, slowdown, completionTime, pa):
        fig = plt.figure()
        res = defaultdict(list)
        {res[key].append(sub[key]) for sub in load_occupied for key in sub}
        # Plot the results
        for i in range(len(models)):
            if pa.objective_disc == pa.objective_slowdown:
                plt.plot(res['cluster_load'], slowdown[i],
                         color=models[i]['color'], marker=models[i]['marker'], label=models[i]['title'])
            elif pa.objective_disc == pa.objective_Ctime:
                plt.plot(res['cluster_load'], completionTime[i],
                         color=models[i]['color'], marker=models[i]['marker'], label=models[i]['title'])

        plt.xlabel("Cluster Load (Percentage)")
        if pa.objective_disc == pa.objective_slowdown:
            plt.ylabel("Average Job Slowdown")
            plt.title("Job Slowdown at different levels of load")
        elif pa.objective_disc == pa.objective_Ctime:
            plt.ylabel("Average Job Completion Time")
            plt.title("Job Completion at different levels of load")
        plt.legend()
        plt.grid()
        plt.show()
        print("Output plotted at", pa.loadVariation_path,
              ' with name ', 'Load_Variation', str(datetime.today()), str(datetime.time(datetime.now())))
        fig.savefig(pa.loadVariation_path +
                    "Load_Variation"+str(datetime.today()) + str(datetime.time(datetime.now())) + pa.figure_extension)

    # Give the rewards, slowdown list and job completion time
    # for the specified number of episodes
    def run_episodes(model, pa, env, agent):
        job_slowdown = []
        job_comption_time = []
        job_reward = []
        episode_list = []

        # Load the model depending on the type of agent
        agent = model['agent']
        if agent == 'DQN':
            model = model['load'].load(model['save_path'], env)
        elif agent == 'DeepRM agent':
            slowdown, completion_time, reward = read_from_file(pa)
            job_slowdown.append(slowdown)
            job_comption_time.append(completion_time)
            job_reward.append(reward)

            return job_reward, job_slowdown, job_comption_time, episode_list
        elif agent == 'A2C' or agent == 'PPO2' or agent == 'ACKTR' or agent == 'TRPO':
            # Create custom policy of custom neurons in the Neural network layers
            class CustomPolicy(FeedForwardPolicy):
                def __init__(self, *args, **kwargs):
                    super(CustomPolicy, self).__init__(*args, **kwargs,
                                                       net_arch=[dict(pi=[pa.network_input_width*pa.network_input_height, 20, pa.network_output_dim],
                                                                      vf=[pa.network_input_width*pa.network_input_height, 20, pa.network_output_dim])],
                                                       feature_extraction="mlp")

            env = make_vec_env(lambda: env, n_envs=1)
            model = model['load'].load(
                model['save_path'], env)

        # Run for multiple episodes
        for episode in range(pa.num_episode):
            cumulated_job_slowdown = []
            cumulated_reward = 0
            cumulated_completion_time = []
            action_list = []
            done = False
            obs = env.reset()
            # Run for multiple sequences
            for seq_idx in range(pa.num_ex):
                while not done:
                    if agent == 'Random':
                        # Take random action
                        action = other_agents.get_random_action(env.job_slot)
                    elif agent == 'SJF':
                        # Take action based on SJF heuristic
                        action = other_agents.get_sjf_action(
                            env.machine, env.job_slot)
                    elif agent == 'Packer':
                        # Take action based on Packer algorithm
                        action = other_agents.get_packer_action(
                            env.machine, env.job_slot)
                    else:
                        # Take action depending on the model loaded
                        action, _states = model.predict(
                            obs, deterministic=False)
                    # take specified action
                    obs, reward, done, info = env.step(action)
                    if isinstance(info, list):
                        info = info[0]
                    if 'Job Slowdown' in info.keys() and 'Completion Time' in info.keys():
                        cumulated_job_slowdown.append(info['Job Slowdown'])
                        cumulated_completion_time.append(
                            info['Completion Time'])
                    if reward != 0:
                        # Append reward at the end of each timestep
                        cumulated_reward += reward
                        action_list.append(action)
                        action_list = []
                    elif reward == 0 and done != True:
                        action_list.append(action)
                    if done == True:
                        break

            # Caluate the mean job slowdown and mean job completion time
            cumulated_job_slowdown = list(flatten(cumulated_job_slowdown))
            cumulated_completion_time = list(
                flatten(cumulated_completion_time))
            episode_list.append(episode + 1)
            if cumulated_completion_time != [] and cumulated_job_slowdown != []:
                job_slowdown.append(np.mean(cumulated_job_slowdown))
                job_comption_time.append(np.mean(cumulated_completion_time))
            else:
                job_comption_time.append(pa.episode_max_length)
                job_slowdown.append(pa.episode_max_length)
            job_reward.append(cumulated_reward)

        return job_reward, job_slowdown, job_comption_time, episode_list

    # Reward list , slowdown list and Completion time list for all models
    reward = [[], [], [], [], [], [], [], [], []]
    slowdown = [[], [], [], [], [], [], [], [], []]
    completionTime = [[], [], [], [], [], [], [], [], []]

    parser = argparse.ArgumentParser(
        description='Load variation')
    parser.add_argument('--num_episodes', type=int, nargs='?', const=1,
                        default=1, help='Maximum number of episodes')
    parser.add_argument('--objective', type=str, nargs='?', const=1, default='Job_Slowdown',
                        help='Objective (Job_Slowdown or Job_Completion_Time)')
    args = parser.parse_args()
    pa = parameters.Parameters()
    # The number of episodes to run
    pa.num_episode = args.num_episodes
    episodes = [n for n in range(pa.num_episode)]

    # Generate graph based on the objective
    if args.objective == 'Job_Slowdown':
        pa.objective_disc = pa.objective_slowdown
        # Sepcify the models
        models = [pa.random_disc, pa.SJF_disc,
                  pa.Packer_disc, pa.DeepRM_agent,
                  pa.A2C_Slowdown, pa.TRPO_Slowdown]
    elif args.objective == 'Job_Completion_Time':
        pa.objective_disc = pa.objective_Ctime
        # Sepcify the models
        models = [pa.random_disc, pa.SJF_disc,
                  pa.Packer_disc, pa.DeepRM_agent,
                  pa.A2C_Ctime, pa.TRPO_Ctime]

    # The job arrival rate list
    cluster_load = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Cluster load in percentage
    cluster_load_percentage = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190]

    # Create list for seperate agents
    slowdown_ep = []
    reward_ep = []
    ct_ep = []
    for i in range(len(cluster_load)):
        list1 = []
        list2 = []
        list3 = []
        for j in range(len(models)):
            list1.append([])
            list2.append([])
            list3.append([])
        slowdown_ep.append(list1)
        reward_ep.append(list2)
        ct_ep.append(list3)

    load_occupied = []
    for i in range(len(cluster_load)):
        # The new job rate determines the cluster load
        pa.new_job_rate = cluster_load[i]
        cluster_values = {}
        cluster_values['simu_len'] = pa.simu_len
        cluster_values['new_job_rate'] = pa.new_job_rate
        # actual loads for which graph will be plotted
        cluster_values['cluster_load'] = cluster_load_percentage[i]
        load_occupied.append(cluster_values)

        # Create Environment
        env = gym.make('deeprm-v0', pa=pa)

        # Run for multiple episodes
        for j in range(len(models)):
            rw, sl, ct, ep = run_episodes(
                models[j], pa, env, models[j]['agent'])
            # Store slowdown, completion time and rewards
            slowdown_ep[i][j].append(np.mean(sl))
            reward_ep[i][j].append(np.mean(rw))
            ct_ep[i][j].append(np.mean(ct))
            print("Done for model", models[j]['agent'],
                  "with job rate", pa.new_job_rate)

    # Calculate the mean over multiple episodes
    for j in range(len(models)):
        for i in range(len(cluster_load)):
            slowdown[j].append(mean(slowdown_ep[i][j]))
            completionTime[j].append(mean(ct_ep[i][j]))

    # Plot the final graph
    plot_load_variation(load_occupied, slowdown, completionTime, pa)
    print("Graph ploted for load variation.")
