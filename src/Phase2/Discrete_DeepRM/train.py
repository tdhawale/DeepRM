from DeepRM.envs.DeepRMEnv import DeepEnv
# import environment
from datetime import datetime
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env
from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import HER
from stable_baselines import GAIL
from stable_baselines import TRPO
from stable_baselines.deepq.policies import MlpPolicy
import gym
import os
from matplotlib.cbook import flatten
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
import argparse
import job_distribution
import parameters
import matplotlib.pyplot as plt
import numpy as np
import other_agents
from statistics import mean
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Plot the slowdown curve for different agents
def plot_slowdown_curve(pa, iterations, env_slowdowns, models):
    fig = plt.figure()
    for i in range(len(models)):
        plt.plot(iterations[i], env_slowdowns[i],
                 color=models[i]['color'], label=models[i]['title'])
    plt.xlabel("Iterations")
    plt.ylabel("Slowdown")
    plt.title('Learning curve')
    plt.legend()
    plt.grid()
    plt.show()
    print("Output plotted at", pa.train_path,
          ' with name ', 'Learningcurve_Slowdown', str(datetime.today()), str(datetime.time(datetime.now())))
    fig.savefig(pa.train_path + 'Learningcurve_Slowdown' + str(datetime.today()) +
                str(datetime.time(datetime.now())) + pa.figure_extension)


# PLot the learning curve
def plot_learning_curve(pa, iterations, env_max_rewards, env_rewards, models):
    fig = plt.figure()
    for i in range(len(models)):
        if models[i]['agent'] == 'A2C':
            plt.plot(iterations[i], env_max_rewards[i],
                     color='plum', label=models[i]['agent'] + 'slowdown Max reward')
            plt.plot(iterations[i], env_rewards[i], color=models[i]['color'],
                     label=models[i]['agent']+'slowdown Mean reward')
        elif models[i]['agent'] == 'PPO2':
            plt.plot(iterations[i], env_max_rewards[i],
                     color='crimson', label=models[i]['agent'] + 'slowdown Max reward')
            plt.plot(iterations[i], env_rewards[i], color=models[i]['color'],
                     label=models[i]['agent'] + 'slowdown Mean reward')
        elif models[i]['agent'] == 'ACKTR':
            plt.plot(iterations[i], env_max_rewards[i],
                     color='deeppink', label=models[i]['agent'] + 'slowdown Max reward')
            plt.plot(iterations[i], env_rewards[i], color=models[i]['color'],
                     label=models[i]['agent'] + 'slowdown Mean reward')
        elif models[i]['agent'] == 'TRPO':
            plt.plot(iterations[i], env_max_rewards[i],
                     color='darkmagenta', label=models[i]['agent'] + 'slowdown Max reward')
            plt.plot(iterations[i], env_rewards[i], color=models[i]['color'],
                     label=models[i]['agent'] + 'slowdown Mean reward')
        elif models[i]['agent'] == 'DQN':
            plt.plot(iterations[i], env_max_rewards[i],
                     color='chartreuse', label=models[i]['agent'] + 'slowdown Max reward')
            plt.plot(iterations[i], env_rewards[i], color=models[i]['color'],
                     label=models[i]['agent']+'slowdown Mean reward')
    plt.xlabel("Iterations")
    plt.ylabel("Rewards")
    plt.title('Learning curve')
    plt.legend()
    plt.grid()
    plt.show()
    print("Output plotted at", pa.train_path,
          ' with name ', 'Learningcurve_Reward', str(datetime.today()), str(datetime.time(datetime.now())))
    fig.savefig(pa.train_path +
                'Learningcurve_Reward'+str(datetime.today())+str(datetime.time(datetime.now())) + pa.figure_extension)


# Give the rewards, slowdown list and job completion time
# for the specified number of episodes
def run_episodes(model, pa, env, agent):
    job_slowdown = []
    job_comption_time = []
    job_reward = []
    episode_list = []
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
                # Take specified action
                obs, reward, done, info = env.step(action)
                if isinstance(info, list):
                    info = info[0]
                if 'Job Slowdown' in info.keys() and 'Completion Time' in info.keys():
                    cumulated_job_slowdown.append(info['Job Slowdown'])
                    cumulated_completion_time.append(info['Completion Time'])
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
        cumulated_completion_time = list(flatten(cumulated_completion_time))
        episode_list.append(episode + 1)
        if cumulated_completion_time != [] and cumulated_job_slowdown != []:
            job_slowdown.append(np.mean(cumulated_job_slowdown))
            job_comption_time.append(np.mean(cumulated_completion_time))
        else:
            job_comption_time.append(pa.episode_max_length)
            job_slowdown.append(pa.episode_max_length)
        job_reward.append(cumulated_reward)

    return job_reward, job_slowdown, job_comption_time, episode_list


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, save_path: str, env: DeepEnv, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.agent = agent
        self.env = env

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            iter = int((self.n_calls / self.check_freq)) + 1
            reward_lst, slowdown_lst, completion_time, episode = run_episodes(model,
                                                                              pa, env, agent)
            # save or update saved model only for best training results.
            # Max possible iterations is time_steps/check_freq.
            mean_reward = np.mean(reward_lst)
            max_reward = max(reward_lst)
            slowdown = np.mean(slowdown_lst)
            ctime = np.mean(completion_time)
            model_slowdowns.append(slowdown)
            model_ctimes.append(ctime)
            model_rewards.append(mean_reward)
            model_max_rewards.append(max_reward)
            model_iterations.append(iter)
            # env.reset()
            if mean_reward >= self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Saving the best model
                self.model.save(self.save_path)
                # print("Saving new best model to {}".format(self.save_path))

            # Reset the environment
            self.env.reset()

            return True


if __name__ == '__main__':
    pa = parameters.Parameters()
    parser = argparse.ArgumentParser(description='Train Agent')
    parser.add_argument('--objective', type=str, nargs='?', const=1, default="Job_Slowdown",
                        help='Job_Slowdown or Job_Completion_Time')
    parser.add_argument('--arival_rate', type=str, nargs='?', const=1, default=0.7,
                        help='Value between [0.1, 0.2, 0.3 ...., 1.0]')
    args = parser.parse_args()

    # Specify the objective and cluster load.
    # The cluster load depends on the job arrival rate
    pa.new_job_rate = args.arival_rate
    pa.objective_disc = args.objective

    env_slowdowns = []
    env_ctimes = []
    env_rewards = []
    env_max_rewards = []
    iterations = []

    # Number of training iterations
    time_steps = pa.time_steps
    check_freq = pa.check_freq

    # Specify the models to train
    if args.objective == pa.objective_slowdown:
        models = [pa.random_disc, pa.SJF_disc, pa.Packer_disc,
                  pa.DQN_Slowdown, pa.PPO2_Slowdown,
                  pa.A2C_Slowdown, pa.TRPO_Slowdown]
    elif args.objective == pa.objective_Ctime:
        models = [pa.random_disc, pa.SJF_disc, pa.Packer_disc,
                  pa.DQN_Ctime, pa.PPO2_Ctime,
                  pa.A2C_Ctime, pa.TRPO_Ctime]

    # Train for different models
    for i in range(len(models)):
        agent = models[i]['agent']
        print("Started for agent", agent)
        log_dir = models[i]['log_dir']
        model_slowdowns = []
        model_ctimes = []
        model_rewards = []
        model_max_rewards = []
        model_iterations = []

        # Create environment
        env = gym.make('deeprm-v0', pa=pa)
        env1 = make_vec_env(lambda: env, n_envs=1)

        # Custom policy with 20 neurons in the hidden layer
        # Input neurons = pa.network_input_width*pa.network_input_height
        # Output Neurons = pa.network_output_dim
        class CustomPolicy(FeedForwardPolicy):
            def __init__(self, *args, **kwargs):
                super(CustomPolicy, self).__init__(*args, **kwargs,
                                                   net_arch=[dict(pi=[pa.network_input_width*pa.network_input_height, 20, pa.network_output_dim],
                                                                  vf=[pa.network_input_width*pa.network_input_height, 20, pa.network_output_dim])],
                                                   feature_extraction="mlp")

        if log_dir != None:
            save_path = models[i]['save_path']
            os.makedirs(log_dir, exist_ok=True)
            # Add some param noise for exploration
            param_noise = AdaptiveParamNoiseSpec(
                initial_stddev=0.1, desired_action_stddev=0.1)
            if agent == 'DQN':
                # Train DQN agent
                model = models[i]['load'](
                    "MlpPolicy", env, verbose=1,
                    learning_rate=1e-3, tensorboard_log=pa.tensorBoard_Logs)
            elif agent == 'TRPO':
                # Train TRPO agent.
                # Here you can specify Custom_policy if
                # you want to customize the neural network
                # parameter
                # Also you can use hyperparameter tuning and
                # change the hyper parameters values
                model = TRPO("MlpPolicy", env, verbose=1,
                             tensorboard_log=pa.tensorBoard_Logs)
            elif agent == 'PPO2':
                # Train PPO2 agent
                # Here you can specify the Custom_policy
                # and change the hyper parameters
                model = models[i]['load'](
                    "MlpPolicy", env1, verbose=0,
                    tensorboard_log=pa.tensorBoard_Logs)
            elif agent == 'A2C':
                # Train A2c agent
                # Here you can specify the Custom_policy
                # and change the hyper parameters
                model = models[i]['load'](
                    "MlpPolicy", env1, verbose=0,
                    learning_rate=1e-3,
                    _init_setup_model=True, policy_kwargs=None, seed=None,
                    tensorboard_log=pa.tensorBoard_Logs)

            # Create callback: check every 100 steps
            callback = SaveOnBestTrainingRewardCallback(
                check_freq=check_freq, log_dir=log_dir, save_path=save_path, env=env)

            # Train the agent
            model.learn(total_timesteps=int(time_steps), callback=callback)
        else:
            # For slowdown curve plot SJF, random and packer
            model = models[i]
            reward_lst, slowdown_lst, completion_time, episode = run_episodes(
                model, pa, env, agent)
            for i in range(int(time_steps/check_freq)):
                model_slowdowns.append(np.mean(slowdown_lst))
                model_ctimes.append(np.mean(completion_time))
                model_rewards.append(np.mean(reward_lst))
                model_iterations.append(i+1)

        env_slowdowns.append(model_slowdowns)
        env_ctimes.append(model_ctimes)
        env_rewards.append(model_rewards)
        env_max_rewards.append(model_max_rewards)
        iterations.append(model_iterations)

    # Plot slowdown curve
    plot_slowdown_curve(pa, iterations, env_slowdowns, models)
    # Plot learning curve
    plot_learning_curve(pa, iterations, env_max_rewards, env_rewards, models)

    print("Done Training")
