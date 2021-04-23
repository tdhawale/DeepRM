from DeepRM.envs.DeepRMEnv import DeepEnv
# import environment
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import parameters
import job_distribution
import argparse
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines import DQN
from stable_baselines import TRPO
from stable_baselines import ACKTR
from stable_baselines.common.env_checker import check_env
from matplotlib.cbook import flatten
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common import make_vec_env
import warnings
from statistics import mean
import other_agents
from datetime import datetime
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# labeling the bar graph
# labeling the bar graph
def autolabel(rects, z):
    i = 0
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.text(rect.get_x() + rect.get_width()/2., height +
                    z[i], '%.1f' % height, ha='center', va='bottom')
        i = i+1


# Give the rewards, slowdown list and job completion time
# for the specified number of episodes

def run_episodes(model, pa, env, agent):

    # Load the model depending on the type of agent
    agent = model['agent']
    if agent == 'DQN':
        model = model['load'].load(model['save_path'], env)
    elif agent == 'A2C' or agent == 'PPO2' or agent == 'ACKTR' or agent == 'TRPO':
        class CustomPolicy(FeedForwardPolicy):
            def __init__(self, *args, **kwargs):
                super(CustomPolicy, self).__init__(*args, **kwargs,
                                                   net_arch=[dict(pi=[4480, 20, 11],
                                                                  vf=[4480, 20, 11])],
                                                   feature_extraction="mlp")

        env = make_vec_env(lambda: env, n_envs=1)
        model = model['load'].load(
            model['save_path'], env)

    job_slowdown = []
    job_comption_time = []
    job_reward = []
    episode_list = []
    withheld_jobs = []
    allocated_jobs = []
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
                if 'Withheld Job' in info.keys():
                    job = info['Withheld Job']
                    job_present = False
                    for existing_job in withheld_jobs:
                        if existing_job.id == job.id:
                            job_present = True
                    if job_present == False:
                        withheld_jobs.append(job)
                if 'Allocated Job' in info.keys():
                    allocated_jobs.append(info['Allocated Job'])
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

    withheld_jobs = list(flatten(withheld_jobs))
    allocated_jobs = list(flatten(allocated_jobs))

    return job_reward, job_slowdown, job_comption_time, episode_list, withheld_jobs, allocated_jobs


if __name__ == '__main__':
    pa = parameters.Parameters()
    pa.new_job_rate = 0.7

    env = gym.make('deeprm-v0', pa=pa)
    env1 = make_vec_env(lambda: env, n_envs=1)

    # Specify the models for execution
    models = [pa.A2C_Slowdown, pa.random_disc]
    # specify the objective
    objectives = [pa.objective_slowdown]

    all_allocated_jobs = []
    for objective in objectives:
        pa.objective_disc = objective
        for model in models:
            env = gym.make('deeprm-v0', pa=pa)
            # Get the details of withheld job
            reward, slowdown, Ctime, episode, withheld_jobs, allocated_jobs = run_episodes(
                model, pa, env, model['agent'])
            all_allocated_jobs.append(allocated_jobs)

    all_len_SD = []
    for i in range(len(models)):
        temp = []
        for k in range(int(pa.max_job_len)):
            temp.append([])
        all_len_SD.append(temp)

    for i in range(len(models)):
        for j in range(len(all_allocated_jobs[i])):
            for k in range(int(pa.max_job_len)):
                if all_allocated_jobs[i][j].len == k+1:
                    all_len_SD[i][k].append(
                        all_allocated_jobs[i][j].job_slowdown)

    fig, ax = plt.subplots()

    A2C_x = A2C_y = A2C_z = Other_x = Other_y = Other_z = []
    for i in range(len(models)):
        x = y = z = ()
        for j in range(len(all_len_SD[i])):
            if all_len_SD[i][j] != []:
                x = x + (j+1,)
                y = y + (mean(all_len_SD[i][j]),)
                z = z + (np.std(all_len_SD[i][j]),)

        if i == 0:
            A2C_x = list(x)
            A2C_y = list(y)
            A2C_z = list(z)
        elif i == 1:
            Other_x = list(x)
            Other_y = list(y)
            Other_z = list(z)

    x = np.arange(len(A2C_x))  # the lenths of generated jobs
    x_ot = np.arange(len(Other_x))
    width = 0.4  # width of the bars
    opacity = 0.8
    fig, ax = plt.subplots()
    if len(models) == 2:
        rects1 = ax.bar(x - width/2, A2C_y, width, yerr=A2C_z, ecolor='r', capsize=4,
                        alpha=opacity, color='g', label='A2C')
        rects2 = ax.bar(x_ot + width/2, Other_y, width, yerr=Other_z, ecolor='c', capsize=4,
                        alpha=opacity, color='m', label='Random')

    # Title and custom x-axis tick labels, etc.

    ax.set_xticks(x)
    ax.set_xticklabels(A2C_x)
    ax.legend()
    plt.xlabel('Job length')
    plt.ylabel('Job Slowdown')
    plt.title('Slowdown versus Job length')
    if len(models) == 2:
        autolabel(rects1, A2C_z)
        autolabel(rects2, Other_z)
    else:
        print("Please add exactly 2 agent names in models list for comparision")
    plt.show()
    fig.savefig('SDVsJobLen.png')
