#File for Joblength VS Slowdown.
from DeepRM.envs.MultiBinaryDeepRM import Env
import gym
import parameters
import job_distribution
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import PPO2, A2C
from stable_baselines.common import make_vec_env
import Script
from statistics import mean

# labeling the bar graph
def autolabel(rects, z):
    i = 0
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.text(rect.get_x() + rect.get_width()/2., height + z[i], '%.1f' % height, ha='center', va='bottom')
        i = i+1


if __name__ == '__main__':
    pa = parameters.Parameters()
    models = [pa.A2C_Slowdown, pa.random]
    pa.new_job_rate = 0.57
    job_sequence_len, job_sequence_size = job_distribution.generate_sequence_work(pa, seed=42)
    env = gym.make('MB_DeepRM-v0', pa=pa)
    env1 = make_vec_env(lambda: env, n_envs=1)
    all_allocated_jobs = []
    for i in range(len(models)):
        pa.objective = models[i]
        log_dir = models[i]['log_dir']
        save_path = None
        model = None
        if models[i]['save_path'] != None:
            save_path = log_dir + models[i]['save_path']
            model = models[i]['agent'].load(save_path, env1)
        episode, reward, slowdown, completion_time, withheld_jobs, allocated_jobs = Script.run_episodes(model, pa, env)
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
                    all_len_SD[i][k].append(all_allocated_jobs[i][j].job_slowdown)

    fig, ax = plt.subplots()

    A2C_x = A2C_y = A2C_z = Other_x = Other_y = Other_z = []
    for i in range(len(models)):
        x = y = z = ()
        for j in range(len(all_len_SD[i])):
            if all_len_SD[i][j] != []:
                x = x + (j+1,)
                y = y + (mean(all_len_SD[i][j]),)
                z = z + (np.std(all_len_SD[i][j]),)

        if i==0:
            A2C_x = list(x)
            A2C_y = list(y)
            A2C_z = list(z)
        elif i==1:
            Other_x = list(x)
            Other_y = list(y)
            Other_z = list(z)           

    x = np.arange(len(A2C_x))  # the lenths of generated jobs
    x_ot = np.arange(len(Other_x))
    width = 0.4  # width of the bars
    opacity = 0.8
    fig, ax = plt.subplots()
    if len(models) == 2:
        rects1 = ax.bar(x - width/2, A2C_y, width, yerr=A2C_z, ecolor=models[0]['yerrcolor'], capsize=4,
            alpha=opacity, color=models[0]['color'], label=models[0]['title'])
        rects2 = ax.bar(x_ot + width/2, Other_y, width, yerr=Other_z, ecolor=models[1]['yerrcolor'], capsize=4,
            alpha=opacity, color=models[1]['color'], label=models[1]['title'])

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
    fig.savefig('workspace/MultiBinary/SDVsJobLen.png')