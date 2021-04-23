# File for with-held jobs.
from DeepRM.envs.MultiBinaryDeepRM import Env
import gym
import parameters
import job_distribution
import matplotlib.pyplot as plt
from stable_baselines import PPO2, A2C
from stable_baselines.common import make_vec_env
import Script

# labeling the bar graph


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    '%.2f' % height, ha='center', va='bottom')
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    '%.2f' % height, ha='center', va='bottom')


def len_withheld_jobs(x, y):
    opacity = 0.8
    agent_plots = []

    agent_plot = plt.bar(x, y,
                         alpha=opacity, color=pa.objective['color'], label=pa.objective['title'])
    agent_plots.append(agent_plot)

    for agent_plot in agent_plots:
        autolabel(agent_plot)

    plt.xlabel('Job length')
    plt.ylabel('Fraction')
    plt.title('Length of withheld jobs')
    plt.xticks(x)
    plt.legend()
    plt.show()
    fig.savefig('workspace/MultiBinary/WithheldJobs.png')


if __name__ == '__main__':
    pa = parameters.Parameters()
    pa.cluster_load = 1.1
    pa.objective = pa.A2C_Slowdown
    pa.simu_len, pa.new_job_rate = job_distribution.compute_simulen_and_arrival_rate(
        pa.cluster_load, pa)
    job_sequence_len, job_sequence_size = job_distribution.generate_sequence_work(
        pa)
    env = gym.make('MB_DeepRM-v0', pa=pa, job_sequence_len=job_sequence_len,
                   job_sequence_size=job_sequence_size)
    env1 = make_vec_env(lambda: env, n_envs=1)

    log_dir = pa.objective['log_dir']
    model = None
    if log_dir != None:
        save_path = log_dir + pa.objective['save_path']
        model = pa.objective['agent'].load(save_path, env1)
    episode, reward, slowdown, completion_time, withheld_jobs, allocated_jobs = Script.run_episodes(
        model, pa, env, job_sequence_len)

    withheld_job_len = []
    x = ()
    y = ()
    for k in range(len(withheld_jobs)):
        withheld_job_len.append(withheld_jobs[k].len)

    fig, ax = plt.subplots()
    if len(withheld_job_len) != 0:
        for i in range(int(pa.max_job_len)):
            x = x + (i+1,)
            y = y + (withheld_job_len.count(i+1)/len(withheld_job_len),)
        len_withheld_jobs(x, y)
        print("Jobs were withheld")
    else:
        print("Jobs were not withheld")
