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
import json
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == '__main__':

    # Read the result from author's implementation
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

    def plot_performance(slowdown_sl, Ctime_ct, models):
        # Divide the plot for job slowdown and for job completion time
        n_groups = 2
        fig, ax = plt.subplots(figsize=(18, 5), dpi=100)
        # fig, ax = plt.subplots(dpi=100)
        index = np.arange(n_groups)
        bar_width = 0.07
        opacity = 0.8
        agent_plots = []
        # Plot the slowdown and completion time for different models
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
        plt.legend(ncol=2)
        ax2 = ax.twinx()
        plt.ylabel("Average job completion time")
        ax2.set_ylim(ax.get_ylim())
        plt.tight_layout()

        plt.tight_layout()
        print("Output plotted at", pa.run_path,
              ' with name ', 'Performance', str(datetime.today()), str(datetime.time(datetime.now())))
        # Save the output
        fig.savefig(pa.run_path + "Performance" + str(datetime.today())
                    + str(datetime.time(datetime.now())) + pa.figure_extension)

    # To label the output graph
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    '%.1f' % height, ha='center', va='bottom')

    # Reward list , slowdown list and Completion time list for
    # all models with objective is minimize job slowdown
    reward_sl = []
    slowdown_sl = []
    Ctime_sl = []

    # Reward list , slowdown list and Completion time list for
    # all models with objective is minimize  job completion time
    reward_ct = []
    slowdown_ct = []
    Ctime_ct = []

    # Get all the parameters
    pa = parameters.Parameters()
    parser = argparse.ArgumentParser(description='Run Scirpt')
    parser.add_argument('--num_episodes', type=int, nargs='?', const=1,
                        default=1, help='Maximum number of episodes')
    parser.add_argument('--new_job_rate', type=float, nargs='?', const=1,
                        default=0.7, help='Specify the job arrival rate')
    args = parser.parse_args()

    # Specifying the job rate. The cluster load is
    # dependant on the job rate
    pa.new_job_rate = args.new_job_rate
    pa.num_episode = args.num_episodes

    # Specify the models for execution
    models = [pa.random_disc, pa.SJF_disc, pa.Packer_disc,
              pa.A2C_Slowdown, pa.A2C_Ctime,
              pa.TRPO_Slowdown, pa.TRPO_Ctime,
              pa.PPO2_Slowdown, pa.PPO2_Ctime,
              pa.DQN_Slowdown, pa.DQN_Ctime]

    # Run for both objective i.e minimize the average job slowdown
    # and minimize the average job completion time
    objectives = [pa.objective_slowdown, pa.objective_Ctime]
    for objective in objectives:
        pa.objective_disc = objective
        for model in models:
            print("Started for agent",
                  model['agent'], 'for objective', objective)
            env = gym.make('deeprm-v0', pa=pa)
            # Run for multiple episodes
            reward, slowdown, Ctime, episode = run_episodes(
                model, pa, env, model['agent'])
            if objective == pa.objective_slowdown:
                reward_sl.append(reward)
                slowdown_sl.append(slowdown)
                Ctime_sl.append(Ctime)
            elif objective == pa.objective_Ctime:
                reward_ct.append(reward)
                slowdown_ct.append(slowdown)
                Ctime_ct.append(Ctime)

    # Plot the performance of all the agents
    plot_performance(slowdown_sl, Ctime_ct, models)
    print("Done")
