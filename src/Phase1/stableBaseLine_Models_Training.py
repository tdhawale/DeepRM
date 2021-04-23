import tensorflow
import pandas as pd
from gym import spaces
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env
from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines.deepq.policies import MlpPolicy
import gym
import os
from random import sample
import argparse
import job_distribution
import parameters
import matplotlib.pyplot as plt
import math
import numpy as np
import operator
from DeepRM.envs.DeepRMEnv import DeepEnv
import warnings
import random
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == '__main__':
    print("------------------------------------------------------------------")
    pa = parameters.Parameters()
    pa.job_wait_queue = 10

    parser = argparse.ArgumentParser(description='Training Agent')
    parser.add_argument('--agent', type=str, nargs='?', const=1, default='A2C', 
                        help='Whether to use A2C, PPO2, or DQN')
    parser.add_argument('--total_train_timesteps', type=int,  nargs='?',
                        const=1, default=25000, help='Number of training steps for the agent')
    parser.add_argument('--objective', type=str, nargs='?', const=1, default=pa.objective_slowdown,
                        help='Path to search spaces for hyperparameter optimization')
    args = parser.parse_args()     

    if args.objective == pa.objective_slowdown:
        pa.objective = pa.objective_slowdown
    else:
        pa.objective = pa.objective_Ctime    

    print("The agent is -> ",args.agent)
    print("The total_train_timesteps is -> ",args.total_train_timesteps)
    print("The objective is -> ",pa.objective)              


    jobsets = job_arrival_rate = []
    print("The model_training_iterations is -> ",pa.model_training_iterations)      
    for i in range(pa.model_training_iterations):
        j = random.randint(1, 100)
        jobsets.append(j)

    job_arrival_rate = [pa.new_job_rate]

    for jobset in jobsets:
        for rate in job_arrival_rate:
            pa.random_seed = jobset
            pa.new_job_rate = rate
            job_sequence_len, job_sequence_size = job_distribution.generate_sequence_work(pa)

            if(args.agent == 'A2C'):
                env = gym.make('deeprm-v0', pa=pa, job_sequence_len=job_sequence_len,
                            job_sequence_size=job_sequence_size)                                
                env1 = make_vec_env(lambda: env, n_envs=1)

                model1 = A2C("MlpPolicy", env1,
                            gamma=0.29,  alpha=0.4, verbose=1, n_steps=5, vf_coef=0.15,
                            ent_coef=0.01, max_grad_norm=0.5, learning_rate=2.5e-4, epsilon=1e-5,
                            lr_schedule='constant', tensorboard_log=pa.tensorBoard_DQN_Logs + args.agent + '/', 
                            _init_setup_model=True, policy_kwargs=None, seed=None, n_cpu_tf_sess=None)

                model1.learn(total_timesteps=args.total_train_timesteps)

                model1.save(pa.model_save_path +"job_scheduling_"+ args.agent +"_"+ args.objective)
            
            elif (args.agent == 'PPO2'):
            
                env = gym.make('deeprm-v0', pa=pa, job_sequence_len=job_sequence_len,
                           job_sequence_size=job_sequence_size)
                env1 = make_vec_env(lambda: env, n_envs=1)

                model2 = PPO2("MlpPolicy", env1, gamma=0.25, learning_rate=1.5e-4, vf_coef=0.25,
                            verbose=1, n_steps=128, ent_coef=0.01, max_grad_norm=0.5, lam=0.52,
                            nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                            tensorboard_log=pa.tensorBoard_DQN_Logs + args.agent + '/', 
                            _init_setup_model=True, policy_kwargs=None,
                            full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)
                model2.learn(total_timesteps=args.total_train_timesteps)
                model2.save(pa.model_save_path +"job_scheduling_"+ args.agent +"_"+ args.objective)

            elif (args.agent == 'DQN'):

                env = gym.make('deeprm-v0', pa=pa, job_sequence_len=job_sequence_len,
                           job_sequence_size=job_sequence_size)

                model3 = DQN("MlpPolicy", env, gamma=0.27, learning_rate=1.5e-4,
                            exploration_initial_eps=0.9,
                            prioritized_replay_alpha=0.2, prioritized_replay_beta0=0.5,
                            prioritized_replay_beta_iters=0.5,
                            prioritized_replay_eps=1e-8, verbose=1,
                            tensorboard_log=pa.tensorBoard_DQN_Logs + args.agent + '/')

                model3.learn(total_timesteps=args.total_train_timesteps)
                model3.save(pa.model_save_path +"job_scheduling_"+ args.agent +"_"+ args.objective)
            
            else:
                print("Invalid Input given")

    print("model save path is -> ",pa.model_save_path +"job_scheduling_"+ args.agent +"_"+ args.objective) 
    print("Training for the agent = " + args.agent + " for the objective = " + args.objective +" is done ")
