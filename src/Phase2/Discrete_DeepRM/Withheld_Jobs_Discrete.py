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
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    '%.2f' % height, ha='center', va='bottom')
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    '%.2f' % height, ha='center', va='bottom')


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
                                                   net_arch=[dict(pi=[pa.network_input_width*pa.network_input_height, pa.network_output_dim, pa.num_nw+1],  # 4480, 20, 11
                                                                  vf=[pa.network_input_width*pa.network_input_height, pa.network_output_dim, pa.num_nw+1])],
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


def len_withheld_jobs(x, y, model, pa):
    opacity = 0.8
    agent_plots = []

    agent_plot = plt.bar(x, y,
                         alpha=opacity, color=model['color'], label=model['title'])
    agent_plots.append(agent_plot)

    for agent_plot in agent_plots:
        autolabel(agent_plot)

    plt.xlabel('Job length')
    plt.ylabel('Fraction')
    plt.title('Fraction of withheld jobs')
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    fig.savefig(pa.withheld_path + "FractionofWithheldJobs" + str(datetime.today())
                + str(datetime.time(datetime.now())) + pa.figure_extension)


def len_generated_jobs(x, y, model, pa):
    opacity = 0.8
    agent_plots = []

    agent_plot = plt.bar(x, y,
                         alpha=opacity, color=model['color'], label=model['title'])
    agent_plots.append(agent_plot)

    for agent_plot in agent_plots:
        autolabel(agent_plot)

    plt.xlabel('Job length')
    plt.ylabel('Fraction')
    plt.title('Fraction of Generated jobs')
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    fig.savefig(pa.withheld_path + "FractionofGeneratedJobs" + str(datetime.today())
                + str(datetime.time(datetime.now())) + pa.figure_extension)


if __name__ == '__main__':
    # Get all the parameters
    pa = parameters.Parameters()
    parser = argparse.ArgumentParser(description='Generating Withheld job')
    parser.add_argument('--new_job_rate', type=float, nargs='?', const=1,
                        default=0.7, help='Specify the job arrival rate')
    args = parser.parse_args()
    pa.new_job_rate = args.new_job_rate

    env = gym.make('deeprm-v0', pa=pa)
    env1 = make_vec_env(lambda: env, n_envs=1)

    # Specify a model for execution
    models = [pa.A2C_Slowdown]
    # specify the objective
    objectives = [pa.objective_slowdown]
    for objective in objectives:
        pa.objective_disc = objective
        for model in models:
            env = gym.make('deeprm-v0', pa=pa)
            # Get the details of withheld job
            reward, slowdown, Ctime, episode, withheld_jobs, allocated_jobs = run_episodes(
                model, pa, env, model['agent'])

    # Determine the length of jobs which were held back
    withheld_job_len = []
    x = ()
    y = ()
    for k in range(len(withheld_jobs)):
        withheld_job_len.append(withheld_jobs[k].len)

    # save graph for witheld jobs
    fig, ax = plt.subplots()
    if len(withheld_job_len) != 0:
        for i in range(int(pa.max_job_len)):
            x = x + (i+1,)
            y = y + (withheld_job_len.count(i+1)/len(withheld_job_len),)
        len_withheld_jobs(x, y, models[0], pa)
        print("Jobs were withheld")
    else:
        print("Jobs were not withheld")

    # Save graph for fraction of jobs generated
    generated_job_len = []
    x = ()
    y = ()
    for k in range(len(allocated_jobs)):
        generated_job_len.append(allocated_jobs[k].len)

    fig, ax = plt.subplots()
    if len(generated_job_len) != 0:
        for i in range(int(pa.max_job_len)):
            x = x + (i+1,)
            y = y + (generated_job_len.count(i+1)/len(generated_job_len),)
        len_generated_jobs(x, y, models[0], pa)
