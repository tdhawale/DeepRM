from DeepRM.envs.DeepRMEnv import DeepEnv
import gym
import tensorflow as tf
import parameters
import job_distribution
import matplotlib.pyplot as plt
import operator
import numpy as np
import math
import matplotlib.pyplot as plt
import parameters
import job_distribution
import argparse
from random import sample
import os
import gym
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines import DQN
import collections
from statistics import mean
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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == '__main__':
    def run(agent, env, objective):

        if agent['agent'] == 'A2C' or agent['agent'] == 'PPO2' or agent['agent'] == 'A2C_Tune_SL':
            env1 = make_vec_env(lambda: env, n_envs=1)
            model = agent['load'].load(agent['save_path'], env1)
        elif agent['agent'] == 'DQN':
            model = agent['load'].load(agent['save_path'], env)

        action_list = []
        obs = env.reset()
        job_slowdown = []
        job_len = []
        withheld_job = []
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
                job_slowdown.append(info['Job Slowdown'])
                job_len.append(info['Job Length'])
                cumulated_completion_time += info['Completion Time']
            if done == True:
                # print("Done")
                break
            if reward != 0:
                cumulated_reward += reward
                action_list.append(action)
                # print("Timestep: ", env.curr_time, "Action: ",
                #       action_list, "Reward: ", reward)
                # Check withheld jobs
                for i in range(len(env.job_slot.slot)):
                    job = env.job_slot.slot[i]
                    if job is not None:
                        for t in range(0, pa.time_horizon - job.len):
                            available_res = env.machine.available_res_slot[t:t + job.len, :]
                            resource_left = available_res - job.resource_requirement
                            if np.all(resource_left[:] >= 0):
                                withheld_job.append(job.len)
                                break
                action_list = []
            if env.curr_time == pa.episode_max_length:
                done = True
            elif reward == 0 and done != True:
                action_list.append(action)

        return cumulated_reward, (cumulated_job_slowdown / len(job_sequence_len)), (cumulated_completion_time / len(job_sequence_len)), job_slowdown, job_len, withheld_job

    slowdown = []
    job_length = []
    withheld = [[], [], []]
    pa = parameters.Parameters()
    pa.job_wait_queue = 10
    pa.objective_disc = pa.objective_slowdown

    models = [pa.Packer_disc, pa.A2C_SL]
    pa.cluster_load = 1.1
    pa.simu_len, pa.new_job_rate = job_distribution.compute_simulen_and_arrival_rate(
        pa.cluster_load, pa)
    job_sequence_len, job_sequence_size = job_distribution.generate_sequence_work(
        pa)
    env = gym.make('deeprm-v0', pa=pa, job_sequence_len=job_sequence_len,
                   job_sequence_size=job_sequence_size)
    env.reset()

    def autolabel(rects, ax):
        #"""Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.text(rect.get_x() + rect.get_width()/2., height,
                        '%.1f' % height, ha='center', va='bottom')

    def plot_slowdown_len(slowdown, job_length, agent1, agent2):
        values = []
        for sl in range(len(slowdown)):
            labels = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [],
                      11: [], 12: [], 13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: []}
            temp = []
            keys = []
            for i in range(len(slowdown[sl])):
                key = job_length[sl][i]
                labels[key].append(slowdown[sl][i])

            for k, v in labels.items():
                if not v:
                    pass
                else:
                    value = mean(v)
                    keys.append(k)
                    temp.append(value)
            values.append(temp)

        # keys = labels.keys()
        x = np.arange(len(keys))  # the label locations
        width = 0.4
        opacity = 0.8
        fig, ax = plt.subplots()
        # rects1 = ax.bar(width, values[0], width, label='SJF')
        rects1 = ax.bar(x - width/2, values[0], width, label=agent1)
        rects2 = ax.bar(x + width/2, values[1], width, label=agent2)
        ax.set_ylabel('Job Slowdown')
        ax.set_xlabel('Job Length')
        ax.set_title('Job Length Vs Slowdown')
        ax.set_xticks(x)
        ax.set_xticklabels(keys)
        ax.legend()
        # autolabel(rects1, ax)
        autolabel(rects1, ax)
        autolabel(rects2, ax)
        fig.tight_layout()
        fig.savefig(pa.figure_path + "JobLength_Vs_Slowdown" +
                    pa.figure_extension)

    for i in range(len(models)):
        reward, sl, Ct, job_slowdown, job_len, withheld_jobs = run(
            models[i], env, pa.objective)
        slowdown.append(job_slowdown)
        job_length.append(job_len)

    # plot_slowdown_len(job_slowdown, job_len)
    plot_slowdown_len(slowdown, job_length,
                      models[0]['title'], models[1]['title'])

    print("JobLength_Vs_Slowdown figure plotted")
