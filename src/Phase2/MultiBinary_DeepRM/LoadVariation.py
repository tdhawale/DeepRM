# File for Load variation.
from DeepRM.envs.MultiBinaryDeepRM import Env
import gym
import matplotlib.pyplot as plt
import parameters
import job_distribution
from stable_baselines import PPO2, A2C
from stable_baselines.common import make_vec_env
import Script
from statistics import mean
import numpy as np
from collections import defaultdict
import numpy
import json


def read_from_file():
    # Specify the path for DeepRM trained agents results
    with open('../Discrete_DeepRM/loadresults.json') as json_file:
        data = json.load(json_file)
        dict = json.loads(data)
    return dict


if __name__ == '__main__':
    pa = parameters.Parameters()
    models = [pa.A2C_Slowdown, pa.PPO2_Slowdown,
              pa.SJF, pa.Packer, pa.random, pa.Hongzimao]
    y_slowdown_readings = []
    new_job_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # run for different values of cluster loads by varying new_job_rates
    for i in range(len(models)):
        pa.objective = models[i]
        log_dir = models[i]['log_dir']
        save_path = None
        Job_arrival_rate_slowdown = []
        Job_arrival_rate_reward = []
        Job_arrival_rate_completion_time = []
        load_occupied = []
        for rate in new_job_rates:
            pa.new_job_rate = rate
            cluster_values = {}
            cluster_values['simu_len'] = pa.simu_len
            cluster_values['new_job_rate'] = pa.new_job_rate
            job_sequence_len, job_sequence_size = job_distribution.generate_sequence_work(
                pa)
            env = gym.make('MB_DeepRM-v0', pa=pa)
            env1 = make_vec_env(lambda: env, n_envs=1)
            # actual loads for which graph will be plotted
            cluster_values['cluster_load'] = pa.cluster_load * 100
            load_occupied.append(cluster_values)

            model = None
            if models[i]['save_path'] != None:
                save_path = log_dir + models[i]['save_path']
                model = models[i]['agent'].load(save_path, env1)

            if models[i] == pa.Hongzimao:
                hong_dict = read_from_file()
                for j in range(len(hong_dict['new_job_rate'])):
                    if hong_dict['new_job_rate'][j] == pa.new_job_rate:
                        slowdown = hong_dict['job_slowdown'][j]
                        completion_time = hong_dict['job_completion_time'][j]
                        reward = 0
            else:
                episode, reward, slowdown, completion_time, withheld_jobs, allocated_jobs = Script.run_episodes(
                    model, pa, env)

            mean_slowdown = np.mean(slowdown)
            mean_reward = np.mean(reward)
            mean_completion_time = np.mean(completion_time)
            Job_arrival_rate_slowdown.append(mean_slowdown)
            Job_arrival_rate_reward.append(mean_reward)
            Job_arrival_rate_completion_time.append(mean_completion_time)

        y_slowdown_readings.append(Job_arrival_rate_slowdown)

    # plot cluster load variation results in graph
    fig = plt.figure()
    res = defaultdict(list)
    {res[key].append(sub[key]) for sub in load_occupied for key in sub}
    for i in range(len(models)):
        plt.plot(res['cluster_load'], y_slowdown_readings[i],
                 color=models[i]['color'], marker=models[i]['marker'], label=models[i]['title'])
    plt.xlabel("Cluster Load(Percentage)")
    plt.ylabel("Average Job Slowdown")
    plt.title("Job slowdown at different levels of load")
    plt.legend()
    plt.grid()
    plt.show()
    fig.savefig('workspace/MultiBinary/ClusterLoadVariation.png')
