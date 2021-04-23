import os
import argparse
import logging
import gym
import numpy as np
from typing import Union
from copy import deepcopy
from ax.service.ax_client import AxClient
import ray
from ray import tune
from ray.tune import report
from ray.tune.suggest.ax import AxSearch
from stable_baselines import A2C, PPO2
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.callbacks import EveryNTimesteps
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines.common.callbacks import BaseCallback
from DeepRM.envs.MultiBinaryDeepRM import Env
import job_distribution
import parameters
from stable_baselines.common import make_vec_env
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt


class OptimizationCallback(BaseCallback):

    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 n_eval_episodes: int = 5,
                 deterministic: bool = False,
                 verbose=0):
        super(OptimizationCallback, self).__init__(verbose)
        self.eval_env = deepcopy(eval_env)
        self.eval_env.reset()
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic

    def _on_step(self):
        sync_envs_normalization(self.training_env, self.eval_env)

        episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                           n_eval_episodes=self.n_eval_episodes,
                                                           render=False,
                                                           deterministic=self.deterministic,
                                                           return_episode_rewards=True)

        mean_reward, std_reward = np.mean(
            episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(
            episode_lengths), np.std(episode_lengths)

        report(
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_ep_length=mean_ep_length,
            std_ep_length=std_ep_length
        )


def main(base_env, pa, args):
    # Reduce the number of Ray warnings that are not relevant here.
    logger = logging.getLogger(tune.__name__)
    logger.setLevel(level=logging.CRITICAL)

    EVAL_EPISODES = args.ray_eval_episodes
    TOTAL_TIMESTEPS = args.total_train_timesteps
    RAY_TUNE_SAMPLES = args.ray_tune_samples
    pa.objective = args.objective
    print("pa objective is ->",pa.objective['save_path'])
    tune_env = deepcopy(base_env)
    tune_monitor = OptimizationCallback(tune_env, EVAL_EPISODES, False)
    monitor_callback = EveryNTimesteps(n_steps=args.report_interval, callback=tune_monitor)
    base_env = make_vec_env(lambda: base_env, n_envs=1)

    def evaluate_objective(config):
        print("config-> ", config)

        if args.agent == A2C or args.agent == PPO2:
            print(args.agent," Agent")
            tune_agent = args.agent
            tune_agent = tune_agent("MlpPolicy", tune_env)
            tune_agent.learn(total_timesteps=TOTAL_TIMESTEPS,callback=monitor_callback)

        else:
            print("Unknown parameters passed in the argument")

    ax_client = AxClient(enforce_sequential_optimization=False)
    if args.agent == PPO2:
        parameters = [
            {"name": "learning_rate", "type": "range",
                "bounds": [1.5e-5, 1e-3]},
            {"name": "gamma", "type": "range", "bounds": [0.20, 1.0]},
            {"name": "vf_coef", "type": "range", "bounds": [0.20, 0.5]},
            {"name": "lam", "type": "range", "bounds": [0.4, 0.9]},
            {"name": "max_grad_norm", "type": "range", "bounds": [0.4, 0.6]},
            {"name": "ent_coef", "type": "range", "bounds": [0.01, 0.02]},
        ]
    elif args.agent == A2C:
        parameters = [
            {"name": "learning_rate", "type": "range",
                "bounds": [1e-05, 0.003]},
            {"name": "gamma", "type": "range", "bounds": [0.99, 1.0]},
            {"name": "vf_coef", "type": "range", "bounds": [0.20, 0.5]},
            {"name": "alpha", "type": "range", "bounds": [0.1, 0.5]},
            {"name": "epsilon", "type": "range", "bounds": [3e-5, 1e-3]},
            {"name": "max_grad_norm", "type": "range", "bounds": [0.3, 7.0]}
        ]
    else:
        print("Invalid argument given for agent")

    ax_client.create_experiment(
        name="tune_RL",
        parameters=parameters,
        objective_name='mean_reward',
        minimize=False)

    # ignore_reinit_error=True
    ray.init(webui_host=pa.localhost, num_cpus=args.ray_cpus, ignore_reinit_error=True)
    tune.run(
        evaluate_objective,
        num_samples=RAY_TUNE_SAMPLES,
        search_alg=AxSearch(ax_client),
        verbose=2
    )

    # get best parameters, retrain agent and log results for best agent
    best_parameters, values = ax_client.get_best_parameters()

    return best_parameters

    
if __name__ == '__main__':
    pa = parameters.Parameters()
    pa.cluster_load = 0.7
    pa.simu_len, pa.new_job_rate = job_distribution.compute_simulen_and_arrival_rate(1 ,pa)
    job_sequence_len, job_sequence_size = job_distribution.generate_sequence_work(pa)
    # parse configuration of experiment
    parser = argparse.ArgumentParser(description='Tune Agent')
    parser.add_argument('--agent', type=str, nargs='?', const=1, default=A2C,
                        help='Whether to use A2C or PPO2')
    parser.add_argument('--total_train_timesteps', type=int,  nargs='?',
                        const=1, default=100000, help='Number of training steps for the agent')
    parser.add_argument('--report_interval', type=int, nargs='?', const=1, default=1000,
                        help='Interval between reportings from callback (in timesteps)')
    parser.add_argument('--ray_eval_episodes', type=int, nargs='?', const=1, default=5,
                        help='Maximum number of episodes for final (deterministic) evaluation')
    parser.add_argument('--ray_tune_samples', type=int, nargs='?', const=1,
                        default=125, help='Number of trials for hyperparameter optimization')
    parser.add_argument('--ray_cpus', type=int, nargs='?', const=1, default=12,
                        help='Number of cpus ray tune will use for the optimization')
    parser.add_argument('--objective', type=str, nargs='?', const=1, default=pa.Tuned_A2C_Slowdown,
                        help='Path to search spaces for hyperparameter optimization')
    parser.add_argument('--logs', type=str, nargs='?', const=1, default=None,
                        help='Path of tensorboard logs for best model after optimization')
    args = parser.parse_args()
    base_env = gym.make('MB_DeepRM-v0', pa=pa, job_sequence_len=job_sequence_len,
                        job_sequence_size=job_sequence_size)
    best_parameters = main(base_env, pa, args)
    base_env1 = make_vec_env(lambda: base_env, n_envs=1)
    log_dir = pa.objective['log_dir']
    save_path = log_dir + pa.objective['save_path']
    best_agent = args.agent("MlpPolicy", base_env1, **best_parameters, tensorboard_log=log_dir)
    best_agent.learn(total_timesteps=args.total_train_timesteps)
    print("Tensorboard logs path -> ",log_dir)
    print("The best parameters were as follows = ",best_parameters)
    best_agent.save(save_path)
    print("Model saved at = ", save_path)