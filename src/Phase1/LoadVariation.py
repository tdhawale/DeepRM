#File for Load variation.
from DeepRM.envs.MultiBinaryDeepRM import Env
import gym
import matplotlib.pyplot as plt
import parameters
import job_distribution
from stable_baselines import PPO2, A2C
from stable_baselines.common import make_vec_env
import Script
from statistics import mean
from collections import defaultdict

if __name__ == '__main__':
    pa = parameters.Parameters()
    models = [pa.A2C_Slowdown, pa.random, pa.SJF, pa.Packer]
    y_slowdown_readings = []
    cluster_load = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                    150, 160, 170, 180, 190]  # aprrox loads which gets recalculated later
    for i in range(len(models)):
        pa.objective = models[i]
        log_dir = models[i]['log_dir']
        save_path = None
        Job_arrival_rate_slowdown = []
        Job_arrival_rate_reward = []
        Job_arrival_rate_completion_time = []
        load_occupied = []
        for rate in cluster_load:
            pa.cluster_load = rate / 100
            pa.simu_len, pa.new_job_rate = job_distribution.compute_simulen_and_arrival_rate(
                cluster_load[-1]/100, pa)
            cluster_values = {}
            cluster_values['simu_len'] = pa.simu_len
            cluster_values['new_job_rate'] = pa.new_job_rate
            job_sequence_len, job_sequence_size = job_distribution.generate_sequence_work(
                pa)
            cluster_values['cluster_occupied'] = pa.cluster_occupied
            # actual loads for which graph will be plotted
            cluster_values['cluster_load'] = pa.cluster_occupied * \
                100 / pa.cluster_capacity
            load_occupied.append(cluster_values)
            env = gym.make('MB_DeepRM-v0', pa=pa, job_sequence_len=job_sequence_len,
                           job_sequence_size=job_sequence_size)
            env1 = make_vec_env(lambda: env, n_envs=1)
            model = None
            if models[i]['save_path'] != None:
                save_path = log_dir + models[i]['save_path']
                model = models[i]['agent'].load(save_path, env1)
            episode, reward, slowdown, completion_time, withheld_jobs, allocated_jobs = Script.run_episodes(
                model, pa, env, job_sequence_len)

            mean_slowdown = min(slowdown)
            mean_reward = min(reward)
            mean_completion_time = min(completion_time)
            Job_arrival_rate_slowdown.append(mean_slowdown)
            Job_arrival_rate_reward.append(mean_reward)
            Job_arrival_rate_completion_time.append(mean_completion_time)

        y_slowdown_readings.append(Job_arrival_rate_slowdown)

    fig = plt.figure()
    res = defaultdict(list)
    {res[key].append(sub[key]) for sub in load_occupied for key in sub}
    for i in range(len(models)):
        plt.plot(res['cluster_load'], y_slowdown_readings[i],
                 color=models[i]['color'], label=models[i]['title'])
    plt.xlabel("Cluster Load(Percentage)")
    plt.ylabel("Average Job Slowdown")
    plt.title("Job slowdown at different levels of load")
    plt.legend()
    plt.grid()
    plt.show()
    fig.savefig('workspace/MultiBinary/ClusterLoadVariation.png')
    # print("Cluster capacity: ", pa.cluster_capacity, ", Job units(total) for various loads: ", load_occupied)