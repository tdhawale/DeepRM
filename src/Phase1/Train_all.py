#File for training agents.
from DeepRM.envs.MultiBinaryDeepRM import Env
import gym
import os
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, A2C
import job_distribution
import parameters
import matplotlib.pyplot as plt
import numpy as np
import Script
from statistics import mean 

class SaveOnBestTrainingRewardCallback(BaseCallback):
  """
  Callback for saving a model (the check is done every ``check_freq`` steps)
  based on the training reward (in practice, we recommend using ``EvalCallback``).
  :param check_freq: (int)
  :param log_dir: (str) Path to the folder where the model will be saved.
  :param verbose: (int)
  """
  def __init__(self, check_freq: int, log_dir: str, save_path: str, verbose=1):
    super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
    self.check_freq = check_freq
    self.log_dir = log_dir
    self.save_path = save_path
    self.best_mean_reward = -np.inf

  def _on_step(self)-> bool:
    if self.n_calls % self.check_freq == 0:
      iter = int((self.n_calls/self.check_freq)) + 1
      episode, reward_lst, slowdown_lst, completion_time, withheld_jobs, allocated_jobs = Script.run_episodes(model, pa, env, job_sequence_len)  
      # save or update saved model only for best training results. Max possible iterations is time_steps/check_freq.
      mean_reward = mean(reward_lst)
      slowdown = mean(slowdown_lst)
      model_slowdowns.append(slowdown)
      model_rewards.append(mean_reward)
      model_max_rewards.append(max(reward_lst))
      model_iterations.append(iter)
      if mean_reward >= self.best_mean_reward:
        self.best_mean_reward = mean_reward
        if self.verbose > 0:
          print("Saving new best model to {}".format(self.save_path))
        self.model.save(self.save_path)

      return True


if __name__ == '__main__':
  pa = parameters.Parameters()
  # models to be trained
  models = [pa.random, pa.SJF, pa.Packer, pa.A2C_Ctime, pa.A2C_Slowdown]
  pa.cluster_load = 0.7
  env_slowdowns = []
  env_rewards = []
  env_max_rewards = []
  check_freq = pa.check_freq
  time_steps = pa.training_time
  iterations = []

  for i in range(len(models)):
    pa.objective = models[i]
    log_dir = models[i]['log_dir']
    
    model_slowdowns = []
    model_rewards = []
    model_max_rewards = []
    model_iterations = []
    
    # Create and wrap the environment
    pa.simu_len, pa.new_job_rate = job_distribution.compute_simulen_and_arrival_rate(1 ,pa)
    job_sequence_len, job_sequence_size = job_distribution.generate_sequence_work(pa)
    env = gym.make('MB_DeepRM-v0', pa=pa, job_sequence_len=job_sequence_len,
                           job_sequence_size=job_sequence_size)
    env1 = make_vec_env(lambda: env, n_envs=1)

    if log_dir != None:
      save_path = log_dir + models[i]['save_path']
      os.makedirs(log_dir, exist_ok=True)
      # Add some param noise for exploration
      param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
      # Because we use parameter noise, we should use a MlpPolicy with layer normalization
      model = models[i]['agent']("MlpPolicy", env1, verbose=1, tensorboard_log=log_dir)

      # Create the callback: check every 100 steps
      callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir, save_path=save_path)
      
      # Train the agent
      model.learn(total_timesteps=int(time_steps), callback=callback)
    else:
      model = models[i]['agent']
      episode, reward_lst, slowdown_lst, completion_time, withheld_jobs, allocated_jobs = Script.run_episodes(model, pa, env, job_sequence_len)
      for i in range(int(time_steps/check_freq)):
        model_slowdowns.append(mean(slowdown_lst))
        model_rewards.append(mean(reward_lst))
        model_iterations.append(i+1)
    env_slowdowns.append(model_slowdowns)
    env_rewards.append(model_rewards)
    env_max_rewards.append(model_max_rewards)
    iterations.append(model_iterations)

  fig2 = plt.figure()
  plt.ylim(0, 15)
  for i in range(len(models)):
    plt.plot(iterations[i], env_slowdowns[i], color=models[i]['color'], label=models[i]['title'])
  plt.xlabel("Iterations")
  plt.ylabel("Slowdown")
  plt.title('Learning curve')
  plt.legend()
  plt.grid()
  plt.show()
  fig2.savefig('workspace/MultiBinary/Learningcurve_Slowdown')

  fig3 = plt.figure()
  for i in range(len(models)):
    if models[i]['save_path'] == 'job_scheduling_A2C_Slowdown':
      plt.plot(iterations[i], env_max_rewards[i],  color='black', label='A2C slowdown Max reward')
      plt.plot(iterations[i], env_rewards[i], color=models[i]['color'], label='A2C slowdown Mean reward')
  plt.xlabel("Iterations")
  plt.ylabel("Rewards")
  plt.title('Learning curve')
  plt.legend()
  plt.grid()
  plt.show()

  fig3.savefig('workspace/MultiBinary/Learningcurve_Reward')