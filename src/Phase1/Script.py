# File for running script.
from DeepRM.envs.MultiBinaryDeepRM import Env
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
import parameters
import job_distribution
from stable_baselines import PPO2, A2C
from stable_baselines.common import make_vec_env
import Otheragents
from statistics import mean

# labeling the bar graph


def autolabel(rects, deviation):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.text(rect.get_x() + rect.get_width()/2., height +
                    deviation[1], '%.2f' % height, ha='center', va='bottom')

# returns the list of average rewards, slowdowns etc., for specified number of episodes episode


def run_episodes(model, pa, env, job_sequence_len):
    episode_list = []
    reward_list = []
    slowdown_list = []
    withheld_jobs = []
    allocated_jobs = []
    completion_time_list = []
    for episode in range(pa.num_episode):
        cumulated_episode_reward = 0
        cumulated_job_slowdown = []
        cumulated_job_completion_time = []
        obs = env.reset()
        done = False
        while not done:
            if model != None:
                action, _states = model.predict(obs, deterministic=False)
            else:
                if pa.objective == pa.random:
                    act = Otheragents.rand_key(env.machine, env.job_slot)
                elif pa.objective == pa.SJF:
                    act = Otheragents.get_sjf_action(env.machine, env.job_slot)
                elif pa.objective == pa.Packer:
                    act = Otheragents.get_packer_action(
                        env.machine, env.job_slot)
                action = []
                for i in range(len(env.job_slot.slot)):
                    if act == i:
                        action.append(1)
                    else:
                        action.append(0)
                action = np.array(action)
            itertime = env.curr_time
            obs, reward, done, info = env.step(action)

            if 'Allocated Job' in info.keys() and info['Allocated Job'] != []:
                for i in range(len(info['Allocated Job'])):
                    job = info['Allocated Job'][i]
                    if job not in allocated_jobs:
                        allocated_jobs.append(job)
                        cumulated_job_slowdown.append(job.job_slowdown)
                        cumulated_job_completion_time.append(
                            job.job_completion_time)
            if 'Withheld Job' in info.keys() and info['Withheld Job'] != []:
                for i in range(len(info['Withheld Job'])):
                    job = info['Withheld Job'][i]
                    if job not in withheld_jobs:
                        withheld_jobs.append(job)

            if done == True:
                break
            if itertime > 0:
                print("Timestep: ", itertime,
                      "Action: ", action, "Reward: ", reward)
            cumulated_episode_reward += reward
            if env.curr_time == pa.episode_max_length:
                done = True
        cumulated_job_completion_time = list(
            flatten(cumulated_job_completion_time))
        cumulated_job_slowdown = list(flatten(cumulated_job_slowdown))

        episode_list.append(episode+1)
        reward_list.append(cumulated_episode_reward)
        if cumulated_job_completion_time != [] and cumulated_job_slowdown != []:
            completion_time_list.append(mean(cumulated_job_completion_time))
            slowdown_list.append(mean(cumulated_job_slowdown))
        else:
            completion_time_list.append(pa.episode_max_length)
            slowdown_list.append(pa.episode_max_length)
    withheld_jobs = list(flatten(withheld_jobs))
    allocated_jobs = list(flatten(allocated_jobs))

    return episode_list, reward_list, slowdown_list, completion_time_list, withheld_jobs, allocated_jobs


if __name__ == '__main__':
    pa = parameters.Parameters()
    models = [pa.A2C_Slowdown, pa.A2C_Ctime,
              pa.Tuned_A2C_Slowdown, pa.random, pa.SJF, pa.Packer]
    pa.cluster_load = 1.3
    pa.simu_len, pa.new_job_rate = job_distribution.compute_simulen_and_arrival_rate(
        pa.cluster_load, pa)
    job_sequence_len, job_sequence_size = job_distribution.generate_sequence_work(
        pa)
    env = gym.make('MB_DeepRM-v0', pa=pa, job_sequence_len=job_sequence_len,
                   job_sequence_size=job_sequence_size)
    env1 = make_vec_env(lambda: env, n_envs=1)

    episodes = []
    rewards = []
    slowdowns = []
    ctimes = []

    for i in range(len(models)):
        pa.objective = models[i]
        log_dir = models[i]['log_dir']
        save_path = None
        model = None
        if models[i]['save_path'] != None:
            save_path = log_dir + models[i]['save_path']
            model = models[i]['agent'].load(save_path, env1)
        episode, reward, slowdown, completion_time, withheld_jobs, allocated_jobs = run_episodes(
            model, pa, env, job_sequence_len)
        episodes.append(episode)
        rewards.append(reward)
        slowdowns.append(slowdown)
        ctimes.append(completion_time)

    # data to plot
    n_groups = 2
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.14
    opacity = 0.8

    for i in range(len(models)):
        mean_values = (mean(slowdowns[i]), mean(ctimes[i]))
        deviation = (np.std(slowdowns[i]), np.std(ctimes[i]))
        agent_plot = plt.bar(index + i*bar_width, mean_values, bar_width, yerr=deviation, ecolor=models[i]['yerrcolor'], capsize=14,
                             alpha=opacity, color=models[i]['color'], label=models[i]['title'])
        autolabel(agent_plot, deviation)

    plt.xlabel('Performance metrics')
    plt.ylabel('Average Job Slowdown')
    plt.title('Performance for different objectives')
    plt.xticks(index + bar_width, ('Slowdown', 'Completion time'))
    plt.legend()
    ax2 = ax.twinx()
    plt.ylabel('Average Job Completion Time')
    ax2.set_ylim(ax.get_ylim())
    plt.tight_layout()
    plt.show()
    fig.savefig('workspace/MultiBinary/Performances.png')
    print("Cluster capacity(units): ", pa.cluster_capacity, ", Job rate:", pa.new_job_rate, ",Simulation length: ",
          pa.simu_len, ", Job units(total): ", pa.cluster_occupied, ", Cluster load: ", pa.cluster_load)
