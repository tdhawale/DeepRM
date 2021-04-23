from DeepRM.envs.DeepRMEnv import DeepEnv
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import parameters
import job_distribution
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines import DQN
import argparse
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env
import randomAgent
import SJF
import packer
import warnings
from statistics import mean
from collections import defaultdict
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == '__main__':
    # Reward list , slowdown list and Completion time list for all models
    reward = [[], [], [], [], [], [], [], [], []]
    slowdown = [[], [], [], [], [], [], [], [], []]
    completionTime = [[], [], [], [], [], [], [], [], []]

    parser = argparse.ArgumentParser(
        description='Load variation')
    parser.add_argument('--num_episodes', type=int, nargs='?', const=1,
                        default=50, help='Maximum number of episodes')
    parser.add_argument('--objective', type=str, nargs='?', const=1, default='Job Slowdown',
                        help='Objective (Job Slowdown or Completion time)')
    args = parser.parse_args()

    def plot_load_variation(load_occupied, slowdown):
        fig = plt.figure()
        res = defaultdict(list)
        {res[key].append(sub[key]) for sub in load_occupied for key in sub}
        for i in range(len(models)):
            plt.plot(res['cluster_load'], slowdown[i],
                     color=models[i]['color'], label=models[i]['title'])
        plt.xlabel("Cluster Load(Percentage)")
        plt.ylabel("Average Job Slowdown")
        plt.title("Job Slowdown at different levels of load")
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(pa.figure_path + "LoadVariation" + pa.figure_extension)

    def run_episode(agent, env):

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
            # elif agent['agent'] == 'DQN':
            #     action, _states = model.predict(obs, deterministic=False)
            else:
                action, _states = model.predict(obs, deterministic=False)

            obs, reward, done, info = env.step(action)
            if bool(info) == True:
                cumulated_job_slowdown += info['Job Slowdown']
                cumulated_completion_time += info['Completion Time']
            if env.curr_time == pa.episode_max_length:
                done = True
            elif reward == 0 and done != True:
                action_list.append(action)
            if reward != 0:
                cumulated_reward += reward
                action_list.append(action)
                # print("Timestep: ", env.curr_time, "Action: ",
                #       action_list, "Reward: ", reward)
                action_list = []
            if done == True:
                # print("Done")
                break

        return cumulated_reward, (cumulated_job_slowdown / len(job_sequence_len)), (cumulated_completion_time / len(job_sequence_len))

    pa = parameters.Parameters()
    pa.num_episode = args.num_episodes
    pa.job_wait_queue = 10
    if args.objective == 'Job Slowdown':
        pa.objective_disc = pa.objective_slowdown
    elif args.objective == 'Job Completion Time':
        pa.objective_disc = pa.objective_Ctime

    episodes = [n for n in range(pa.num_episode)]

    models = [pa.random_disc, pa.SJF_disc, pa.Packer_disc,
              pa.DQN_SL, pa.DQN_CT, pa.PPO2_SL,
              pa.PPO2_CT, pa.A2C_SL, pa.A2C_CT]

    cluster_load = [10, 20, 30, 40, 50, 60, 70, 80, 90,
                    100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

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

    for episode in range(pa.num_episode):
        load_occupied = []
        for i in range(len(cluster_load)):
            rate = cluster_load[i]
        # for rate in cluster_load:
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
            # env = DeepRMEnv.Env(pa, job_sequence_len=job_sequence_len,
            #                     job_sequence_size=job_sequence_size)
            env = gym.make('deeprm-v0', pa=pa, job_sequence_len=job_sequence_len,
                           job_sequence_size=job_sequence_size)
            for j in range(len(models)):
                rw, sl, ct = run_episode(models[j], env)
                slowdown_ep[i][j].append(sl)
                reward_ep[i][j].append(rw)
                ct_ep[i][j].append(ct)
                print("Done for episode", episode,
                      "with seed", pa.random_seed,
                      "for model", models[j]['agent'],
                      "with rate", rate)

    for j in range(len(models)):
        for i in range(len(cluster_load)):
            slowdown[j].append(mean(slowdown_ep[i][j]))

    plot_load_variation(load_occupied, slowdown)
    print("Graph ploted for load variation.")
