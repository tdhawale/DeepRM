import numpy as np
import math
import job_distribution
from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import ACKTR
from stable_baselines import TRPO


class Parameters:
    def __init__(self):

        self.output_filename = 'data/tmp'
        self.num_episode = 1
        self.num_epochs = 10        # number of training epochs
        self.simu_len = 50          # length of the busy cycle that repeats itself
        self.num_ex = 1             # number of sequences

        self.output_freq = 10          # interval for output and store parameters

        self.num_seq_per_batch = 20    # number of sequences to compute baseline
        self.episode_max_length = 2000  # enforcing as artificial terminal

        self.num_res = 2               # number of resources in the system
        self.num_nw = 10               # maximum allowed number of work in the queue

        self.time_horizon = 20         # number of time steps in the graph
        self.max_job_len = 15          # maximum duration of new jobs
        self.res_slot = 10             # maximum number of available resource slots
        self.max_job_size = 10         # maximum resource request of new work

        self.backlog_size = 60         # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.job_num_cap = 40          # maximum number of distinct colors in current work graph

        self.new_job_rate = 0.7       # lambda in new job arrival Poisson Process

        self.discount = 1           # discount factor

        self.time_steps = 10000000
        self.check_freq = 1000

        # distribution for new job arrival
        self.dist = job_distribution.Dist(
            self.num_res, self.max_job_size, self.max_job_len)

        # graphical representation
        # such that it can be converted into an image
        assert self.backlog_size % self.time_horizon == 0
        self.backlog_width = int(
            math.ceil(self.backlog_size / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action

        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 0.001          # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy
        self.batch_size = 10
        self.evaluate_policy_name = "SJF"
        self.objective_slowdown = "Job_Slowdown"
        self.objective_Ctime = "Job_Completion_Time"
        self.objective_disc = self.objective_slowdown

        # Path for saving the trained model.
        self.model_save_path = 'models/'
        # Path for storing the training results.
        self.train_path = 'output/train/'
        # Path for storing the withheld generated graph
        self.withheld_path = 'output/WithheldJobs/'
        # Path for storing the performance comparision graph.
        self.run_path = 'output/run/'
        # Path for storing the job distribution
        self.job_generation_path = 'output/JobDistribution/'
        # Path for storing the load variation graph.
        self.loadVariation_path = 'output/loadVariation/'
        self.figure_extension = '.png'
        # Path for storing the tensor logs.
        self.tensorBoard_Logs = 'tensorboard/'

        self.DeepRM_agent = {'agent': 'DeepRM agent',
                             'save_path': None,
                             'log_dir': None,
                             'color': 'Purple',
                             'yerrcolor': 'Violet',
                             'marker': "P",
                             'title': 'DeepRM agent',
                             'load': None}
        self.random_disc = {'agent': 'Random',
                            'save_path': None,
                            'color': 'r',
                            'log_dir': None,
                            'yerrcolor': 'Gold',
                            'marker': "D",
                            'title': 'Random Agent',
                            'load': None}
        self.SJF_disc = {'agent': 'SJF',
                         'save_path': None,
                         'color': 'orange',
                         'log_dir': None,
                         'yerrcolor': 'Hotpink',
                         'marker': "o",
                         'title': 'SJF',
                         'load': None}
        self.Packer_disc = {'agent': 'Packer',
                            'save_path': None,
                            'color': 'g',
                            'log_dir': None,
                            'yerrcolor': 'teal',
                            'marker': "^",
                            'title': 'Packer',
                            'load': None}

        self.A2C_Slowdown = {'agent': 'A2C',
                             'save_path': self.model_save_path + "job_scheduling_" + "A2C_Slowdown",
                             'yerrcolor': 'purple',
                             'marker': "s",
                             'log_dir': self.model_save_path,
                             'color': 'c',
                             'title': 'Discrete - A2C agent trained for SL ',
                             'load': A2C}

        self.A2C_Ctime = {'agent': 'A2C',
                          'save_path': self.model_save_path + "job_scheduling_" + "A2C_Ctime",
                          'yerrcolor': 'deeppink',
                          'marker': "v",
                          'log_dir': self.model_save_path,
                          'color': 'b',
                          'title': 'Discrete - A2C agent trained for CT',
                          'load': A2C}

        self.TRPO_Slowdown = {'agent': 'TRPO',
                              'save_path': self.model_save_path + "job_scheduling_" + "TRPO_Slowdown",
                              'yerrcolor': 'darkseagreen',
                              'marker': "h",
                              'log_dir': self.model_save_path,
                              'color': 'k',
                              'title': 'Discrete - TRPO agent trained for SL',
                              'load': TRPO}

        self.TRPO_Ctime = {'agent': 'TRPO',
                           'save_path': self.model_save_path + "job_scheduling_" + "TRPO_Ctime",
                           'yerrcolor': 'greenyellow',
                           'marker': "p",
                           'log_dir': self.model_save_path,
                           'color': 'm',
                           'title': 'Discrete - TRPO agent trained for CT',
                           'load': TRPO}

        self.PPO2_Slowdown = {'agent': 'PPO2',
                              'save_path': self.model_save_path + "job_scheduling_" + "PPO2_Slowdown",
                              'yerrcolor': 'khaki',
                              'marker': "x",
                              'log_dir': self.model_save_path,
                              'color': 'y',
                              'title': 'Discrete - PPO2 agent trained for SL',
                              'load': PPO2}

        self.PPO2_Ctime = {'agent': 'PPO2',
                           'save_path': self.model_save_path + "job_scheduling_" + "PPO2_Ctime",
                           'yerrcolor': 'gold',
                           'marker': "d",
                           'log_dir': self.model_save_path,
                           'color': 'indigo',
                           'title': 'Discrete - PPO2 agent trained for CT',
                           'load': PPO2}

        self.DQN_Slowdown = {'agent': 'DQN',
                             'save_path': self.model_save_path + "job_scheduling_" + "DQN_Slowdown",
                             'yerrcolor': 'tomato',
                             'marker': "1",
                             'log_dir': self.model_save_path,
                             'color': 'coral',
                             'title': 'Discrete - DQN agent trained for SL',
                             'load': DQN}

        self.DQN_Ctime = {'agent': 'DQN',
                          'save_path': self.model_save_path + "job_scheduling_" + "DQN_Ctime",
                          'yerrcolor': 'brown',
                          'marker': "11",
                          'log_dir': self.model_save_path,
                          'color': 'brown',
                          'title': 'Discrete - DQN agent trained for CT',
                          'load': DQN}

        # Tuned agent

        self.A2C_Slowdown_tuned = {'agent': 'A2C',
                                   'save_path': self.model_save_path + "job_scheduling_" + "A2C_" + self.objective_slowdown + "_tuned",
                                   'yerrcolor': 'purple',
                                   'marker': "s",
                                   'log_dir': self.model_save_path,
                                   'color': 'c',
                                   'title': 'Discrete - A2C agent tuned for SL ',
                                   'load': A2C}

        self.A2C_Ctime_tuned = {'agent': 'A2C',
                                'save_path': self.model_save_path + "job_scheduling_" + "A2C_" + self.objective_Ctime + "_tuned",
                                'yerrcolor': 'deeppink',
                                'marker': "v",
                                'log_dir': self.model_save_path,
                                'color': 'b',
                                'title': 'Discrete - A2C agent tuned for CT',
                                'load': A2C}

        self.TRPO_Slowdown_tuned = {'agent': 'TRPO',
                                    'save_path': self.model_save_path + "job_scheduling_" + "TRPO_" + self.objective_slowdown + "_tuned",
                                    'yerrcolor': 'darkseagreen',
                                    'marker': "h",
                                    'log_dir': self.model_save_path,
                                    'color': 'k',
                                    'title': 'Discrete - TRPO agent tuned for SL',
                                    'load': TRPO}

        self.TRPO_Ctime_tuned = {'agent': 'TRPO',
                                 'save_path': self.model_save_path + "job_scheduling_" + "TRPO_" + self.objective_Ctime + "_tuned",
                                 'yerrcolor': 'greenyellow',
                                 'marker': "p",
                                 'log_dir': self.model_save_path,
                                 'color': 'm',
                                 'title': 'Discrete - TRPO agent tuned for CT',
                                 'load': TRPO}

        self.PPO2_Slowdown_tuned = {'agent': 'PPO2',
                                    'save_path': self.model_save_path + "job_scheduling_" + "PPO2_" + self.objective_slowdown + "_tuned",
                                    'yerrcolor': 'khaki',
                                    'marker': "x",
                                    'log_dir': self.model_save_path,
                                    'color': 'y',
                                    'title': 'Discrete - PPO2 agent tuned for SL',
                                    'load': PPO2}

        self.PPO2_Ctime_tuned = {'agent': 'PPO2',
                                 'save_path': self.model_save_path + "job_scheduling_" + "PPO2_" + self.objective_Ctime + "_tuned",
                                 'yerrcolor': 'gold',
                                 'marker': "d",
                                 'log_dir': self.model_save_path,
                                 'color': 'indigo',
                                 'title': 'Discrete - PPO2 agent tuned for CT',
                                 'load': PPO2}

        self.DQN_Slowdown_tuned = {'agent': 'DQN',
                                   'save_path': self.model_save_path + "job_scheduling_" + "DQN_" + self.objective_slowdown + "_tuned",
                                   'yerrcolor': 'tomato',
                                   'marker': "1",
                                   'log_dir': self.model_save_path,
                                   'color': 'coral',
                                   'title': 'Discrete - DQN agent tuned for SL',
                                   'load': DQN}

        self.DQN_Ctime_tuned = {'agent': 'DQN',
                                'save_path': self.model_save_path + "job_scheduling_" + "DQN_" + self.objective_Ctime + "_tuned",
                                'yerrcolor': 'brown',
                                'marker': "11",
                                'log_dir': self.model_save_path,
                                'color': 'brown',
                                'title': 'Discrete - DQN agent tuned for CT',
                                'load': DQN}

    #  calculating the dimensions of the observation space.

    def compute_dependent_parameters(self):
        # such that it can be converted into an image
        assert self.backlog_size % self.time_horizon == 0
        self.backlog_width = self.backlog_size / self.time_horizon
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job
        self.network_input_width = int(self.network_input_width)
        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + \
            1  # + 1 for backlog indicator + self.num_nw) + 1  # + 1 for backlog indicator
        self.network_output_dim = self.num_nw + 1  # + 1 for void action
