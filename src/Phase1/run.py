from DeepRM.envs.DeepRMEnv import DeepEnv
import gym
import tensorflow as tf
import operator
import numpy as np
import math
import matplotlib.pyplot as plt
import parameters
import job_distribution
import argparse
from random import sample
import os
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env
from gym import spaces
import pandas as pd
from stable_baselines.common import make_vec_env
import randomAgent
import SJF
import packer
import warnings
import random
from statistics import mean
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=Warning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == '__main__':
    # Reward list , slowdown list and Completion time list for all models with objective is minimize job slowdown
    reward_sl = [[], [], [], [], [], [], [], [], []]
    slowdown_sl = [[], [], [], [], [], [], [], [], []]
    Ctime_sl = [[], [], [], [], [], [], [], [], []]
    # Reward list , slowdown list and Completion time list for all models with objective is minimize  job completion time
    reward_ct = [[], [], [], [], [], [], [], [], []]
    slowdown_ct = [[], [], [], [], [], [], [], [], []]
    Ctime_ct = [[], [], [], [], [], [], [], [], []]

    parser = argparse.ArgumentParser(
        description='Load variation')
    parser.add_argument('--num_episodes', type=int, nargs='?', const=1,
                        default=50, help='Maximum number of episodes')
    args = parser.parse_args()

    pa = parameters.Parameters()
    pa.num_episode = args.num_episodes
    pa.job_wait_queue = 10
    episodes = [n for n in range(pa.num_episode)]

    def run_episode(agent, env, objective):
        if agent['agent'] == 'A2C' or agent['agent'] == 'PPO2':
            env1 = make_vec_env(lambda: env, n_envs=1)
            model = agent['load'].load(agent['save_path'], env1)
        elif agent['agent'] == 'DQN':
            model = agent['load'].load(agent['save_path'], env)

        action_list = []
        obs = env.reset()
        cumulated_job_slowdown = 0
        cumulated_reward = 0
        cumulated_completion_time = 0
        done = False

        while not done:
            if agent['agent'] == 'RandomAgent':
                action = randomAgent.agent(env.job_slot)
            elif agent['agent'] == 'SJF':
                action = SJF.agent(env.machine, env.job_slot)
            elif agent['agent'] == 'Packer':
                action = packer.agent(env.machine, env.job_slot)
            elif agent['agent'] == 'DQN':
                action, _states = model.predict(obs, deterministic=False)
            else:
                action, _states = model.predict(obs, deterministic=False)

            obs, reward, done, info = env.step(action)
            if bool(info) == True:
                cumulated_job_slowdown += info['Job Slowdown']
                cumulated_completion_time += info['Completion Time']
            if done == True:
                # print("Done")
                break
            if reward != 0:
                cumulated_reward += reward
                action_list.append(action)
                # print("Timestep: ", env.curr_time, "Action: ",
                #       action_list, "Reward: ", reward)
                action_list = []
            if env.curr_time == pa.episode_max_length:
                done = True
            elif reward == 0 and done != True:
                action_list.append(action)

        return cumulated_reward, (cumulated_job_slowdown / len(job_sequence_len)), (cumulated_completion_time / len(job_sequence_len))

    models = [pa.random_disc, pa.SJF_disc, pa.Packer_disc,
              pa.DQN_SL, pa.DQN_CT, pa.PPO2_SL,
              pa.PPO2_CT, pa.A2C_SL, pa.A2C_CT]
    objectives = [pa.objective_slowdown, pa.objective_Ctime]

    for episode in episodes:
        pa.random_seed = random.randint(1, 1000)
        job_sequence_len, job_sequence_size = job_distribution.generate_sequence_work(
            pa)

        for objective in objectives:
            pa.objective_disc = objective
            env = gym.make('deeprm-v0', pa=pa, job_sequence_len=job_sequence_len,
                           job_sequence_size=job_sequence_size)
            env.reset()
            i = 0
            for model in models:
                reward, slowdown, Ctime = run_episode(model, env, objective)
                if objective == pa.objective_slowdown:
                    reward_sl[i].append(reward)
                    slowdown_sl[i].append(slowdown)
                    Ctime_sl[i].append(Ctime)
                elif objective == pa.objective_Ctime:
                    reward_ct[i].append(reward)
                    slowdown_ct[i].append(slowdown)
                    Ctime_ct[i].append(Ctime)
                i += 1
                print("Done for episode", episode,
                      "with seed", pa.random_seed,
                      "for model", model['agent'],
                      "with objective", objective)
    print("Done")


# Plot job slowdown or completion time per episode
def plot_slowdown(episodes, slowdown_sl, models):
    fig = plt.figure(figsize=(20, 5), dpi=100)
    for i in range(len(models)):
        plt.plot(episodes, slowdown_sl[i],
                 color=models[i]['color'], label=models[i]['title'])
    plt.xlabel("Number of episodes")
    plt.ylabel("Average Job Slowdown per episodes")
    plt.title("Average Job Slowdown per episode for Stable Baselines")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.show()
    fig.savefig(pa.figure_path + "JobSlowdown" +
                pa.figure_extension, bbox_inches='tight')


def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%.1f' % height, ha='center', va='bottom')


def plot_performance(slowdown_sl, Ctime_ct, models):
    n_groups = 2
    # create plot
    fig, ax = plt.subplots(figsize=(20, 5), dpi=100)
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.8
    agent_plots = []
    for i in range(len(models)):
        mean_values = (mean(slowdown_sl[i]), mean(Ctime_ct[i]))
        deviation = (np.std(slowdown_sl[i]), np.std(Ctime_ct[i]))
        agent_plot = plt.bar(index + i * bar_width, mean_values, bar_width,
                             yerr=deviation,
                             ecolor='k',
                             capsize=14,
                             alpha=opacity,
                             color=models[i]['color'],
                             label=models[i]['title'])
        agent_plots.append(agent_plot)
    for agent_plot in agent_plots:
        autolabel(agent_plot, ax)
    plt.xlabel('Performance metrics')
    plt.ylabel('Average job slowdown')
    plt.title('Performance for different objective')
    plt.xticks(index + 4 * bar_width, ('Average job slowdown',
                                       'Average job completion time'))
    plt.legend()
    ax2 = ax.twinx()
    plt.ylabel("Average job completion time")
    ax2.set_ylim(ax.get_ylim())
    plt.tight_layout()

    plt.tight_layout()
    fig.savefig(pa.figure_path + "Performance" + pa.figure_extension)


plot_slowdown(episodes, slowdown_sl, models)
plot_performance(slowdown_sl, Ctime_ct, models)

print("Performance graphs plotted")
