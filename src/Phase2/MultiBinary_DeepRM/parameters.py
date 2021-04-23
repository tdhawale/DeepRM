# File for parameters.
import job_distribution
from stable_baselines import A2C, PPO2, ACKTR
import math


class Parameters:
    # intializing parameters required for env
    def __init__(self):

        self.simu_len = 100  # for complete 1.9 cluster load
        # length of the busy cycle that repeats itself
        self.num_ex = 10
        # number of sequences
        self.output_freq = 10
        # interval for output and store parameters
        self.num_seq_per_batch = 10
        # number of sequences to compute baseline ???
        self.episode_max_length = 2000
        # enforcing an artificial terminal - no of feasable training episodes
        self.num_res = 2
        # number of resources in the system - CPU,Memory
        self.num_nw = 5
        # maximum allowed number of work/jobs in the queue. M
        self.time_horizon = 20
        # number of time steps in the graph
        self.max_job_len = self.time_horizon * 0.75
        # maximum duration of new jobs
        self.res_slot = 20
        # maximum number of available resource slots
        self.max_job_size = 10
        # maximum resource request of new work/Jobs
        self.backlog_size = 60
        # backlog queue size in waiting queue to go in M.
        self.max_track_since_new = 10
        # track how many time steps since last new jobs
        self.job_num_cap = 40
        # maximum number of distinct colors in current work graph . not required
        self.new_job_rate = 1
        assert self.backlog_size % self.time_horizon == 0
        # penalty for delaying things in the current work screen
        self.penalty = -1
        self.verbose = 0 # set to 1 for printing actions taken wrt each timestep
        self.batch_size = 10
        # random number seed
        self.random_seed = 42
        self.num_episode = 10
        self.job_small_chance = 0.8
        self.cluster_capacity = (
            self.res_slot ** self.num_res) * self.time_horizon
        self.cluster_load = 1.3
        self.cluster_occupied = 0
        # 0,1,3 for any one dominant and 2 for both dominant resources, 4,5, etc... for all other
        self.dominant_res = 3
        self.check_freq = 100
        self.training_time = 1e5
        self.move_on_count = 4
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
        
        self.dist = job_distribution.Dist(self.num_res, self.max_job_size, self.max_job_len)

        # MB
        self.Hongzimao = {'agent': None, 'save_path': None,
                          'log_dir': None, 'color': 'Purple', 'yerrcolor': 'Violet', 'marker': "P", 'title': 'DeepRM'}    
            
        self.A2C_Ctime = {'agent': A2C, 'save_path': 'job_scheduling_A2C_Ctime',
                          'log_dir': "workspace/MultiBinary/tensor_A2C_Ctime/", 'color': 'Red', 'yerrcolor': 'Brown', 'title': 'A2C Ctime agent'}
        self.A2C_Slowdown = {'agent': A2C, 'save_path': 'job_scheduling_A2C_Slowdown',
                             'log_dir': "workspace/MultiBinary/tensor_A2C_Slowdown/", 'color': 'SkyBlue', 'yerrcolor': 'Blue',  'marker': "s", 'title': 'A2C Slowdown agent'}
        self.PPO2_Slowdown = {'agent': PPO2, 'save_path': 'job_scheduling_PPO2_Slowdown', 'log_dir': "workspace/MultiBinary/tensor_PPO2_Slowdown/",
                     'color': 'Green', 'yerrcolor': 'DarkGreen', 'marker': "x", 'title': 'PPO2 Slowdown agent'}
        self.PPO2_Ctime = {'agent': PPO2, 'save_path': 'job_scheduling_PPO2_Ctime', 'log_dir': "workspace/MultiBinary/tensor_PPO2_Ctime/",
                     'color': 'chocolate', 'yerrcolor': 'brown', 'title': 'PPO2 Completion time agent'}
        self.ACKTR_Slowdown = {'agent': ACKTR, 'save_path': 'job_scheduling_ACKTR_Slowdown', 'log_dir': "workspace/MultiBinary/tensor_ACKTR_Slowdown/",
                     'color': 'blue', 'yerrcolor': 'blue', 'title': 'ACKTR Slowdown agent'}
        self.ACKTR_Ctime = {'agent': ACKTR, 'save_path': 'job_scheduling_ACKTR_Ctime', 'log_dir': "workspace/MultiBinary/tensor_ACKTR_Ctime/",
                     'color': 'red', 'yerrcolor': 'brown', 'title': 'ACKTR Completion time agent'}
                     
        self.random = {'agent': None, 'save_path': None, 'log_dir': None,
                       'color': 'Yellow', 'yerrcolor': 'Gold', 'marker': "D", 'title': 'Random agent'}
        self.SJF = {'agent': None, 'save_path': None, 'log_dir': None,
                    'color': 'pink', 'yerrcolor': 'Hotpink', 'marker': "o", 'title': 'SJF agent'}
        self.Packer = {'agent': None, 'save_path': None, 'log_dir': None,
                       'color': 'lime', 'yerrcolor': 'teal', 'marker': "^", 'title': 'Packer agent'}

        self.objective = self.A2C_Slowdown

        self.figure_extension = '.png'
