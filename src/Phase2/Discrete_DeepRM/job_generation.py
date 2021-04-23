# ---------------------------------------------------
# # Understanding the job generation to determine
# the cluster load
# ---------------------------------------------------
from DeepRM.envs.DeepRMEnv import DeepEnv
import numpy as np
import matplotlib.pyplot as plt
import parameters
import job_distribution
import warnings
from statistics import mean
import other_agents
from datetime import datetime
import matplotlib.cm as cm
import gym


if __name__ == '__main__':
    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.num_nw = 10

    # Check the distribution of jobs for different job arrival rates.
    for new_job_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        pa.new_job_rate = new_job_rate
        # generate sequence for different job rates
        job_sequence_len, job_sequence_size = job_distribution.generate_sequence_work(
            pa, 42)

        # Plot the job Distribution
        fig = plt.figure()
        job_sequence_len_plot = []
        jobs = [i for i in range(pa.simu_len * pa.num_ex)]
        index_start = 0
        index_finish = pa.simu_len

        non_zero_list = []
        count_index = [l for l in range(pa.num_ex)]
        colors = cm.rainbow(np.linspace(0, 1, pa.num_ex))

        # Plotting the bar graph showing the distribution of jobs
        for j in range(pa.num_ex):
            jobs = [index for index in range(index_start, index_finish)]
            plt.bar(jobs, job_sequence_len[j], color=colors[j], width=5)
            index_start = index_finish
            index_finish = index_finish + pa.simu_len
            non_zero_list.append(np.count_nonzero(job_sequence_len[j]))
        plt.xlabel('Jobs')
        plt.ylabel('Job Length')
        plt.title('Job Distribution')
        fig.savefig(pa.job_generation_path + 'script_jobs_' + str(pa.new_job_rate) + '_' + str(pa.simu_len)+'_' + str(datetime.today()) + '_' + str(datetime.time(datetime.now())) +
                    pa.figure_extension)

        # Plot the number of Non Zero Jobs
        fig = plt.figure()
        plt.bar(count_index, non_zero_list)
        plt.xlabel('Sequence')
        plt.ylabel('Non zero length jobs')
        plt.title('Number of non zero length jobs in each sequence')
        fig.savefig(pa.job_generation_path + 'script_non_zero_jobs_' + str(pa.new_job_rate)+'_' + str(pa.simu_len)+'_' + str(datetime.today()) + '_' + str(datetime.time(datetime.now())) +
                    pa.figure_extension)

        # Plotting the slowdown for the generated jobs for SJF and Packer agents
        env = gym.make('deeprm-v0', pa=pa, nw_len_seqs=job_sequence_len,
                       nw_size_seqs=job_sequence_size,
                       render=False, repre='image', end='all_done')
        done = False
        models = ['SJF', 'Packer']

        job_slowdown = []
        job_comption_time = []
        job_reward = []
        episode_list = []
        for model in models:
            cumulated_job_slowdown = []
            cumulated_reward = 0
            cumulated_completion_time = []
            ob = env.reset()
            done = False
            info = {}
            while not done:
                if model == 'SJF':
                    action = other_agents.get_sjf_action(
                        env.machine, env.job_slot)
                else:
                    action = other_agents.get_packer_action(
                        env.machine, env.job_slot)
                ob, reward, done, info = env.step(action)
                if 'Job Slowdown' in info.keys() and 'Completion Time' in info.keys():
                    cumulated_job_slowdown.append(info['Job Slowdown'])
                    cumulated_completion_time.append(info['Completion Time'])
                if reward != 0:
                    cumulated_reward += reward

            job_slowdown.append(np.mean(cumulated_job_slowdown))
            job_comption_time.append(np.mean(cumulated_completion_time))
            job_reward.append(cumulated_reward)

        fig = plt.figure()
        plt.bar(models, job_slowdown, color=['r', 'y'])
        plt.xlabel('Heuristics')
        plt.ylabel('Average Job Slowdown')
        plt.title('Job Slowdown')
        fig.savefig(pa.job_generation_path + 'script_slowdown_' + str(pa.new_job_rate)+'_' + str(pa.simu_len)+'_' + str(pa.simu_len)+'_' + str(datetime.time(datetime.now())) +
                    pa.figure_extension)
    print("Done")
