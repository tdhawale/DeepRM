# --------------------------------------------------------------------------------
# Environment for scheduling jobs to cluster with resources CPU and memory.
# We take Discrete actions in which the agent selects one job from the job slots
# and assigns it to the machine. In order to allow the agent to take multiple
# action, the current time is frozen until the agent takes an invalid or void
# action.
#
# The implementation is based with reference to
# 'http://people.csail.mit.edu/hongzi/content/publications/DeepRM-HotNets16.pdf'
# -------------------------------------------------------------------------------

import numpy as np
import math
import matplotlib.pyplot as plt
import gym
from gym import spaces

# Create custom environment for Scheduling.


class DeepEnv(gym.Env):
    # We pass a dictionary of parameters along with job requirements and also we set
    # the end type to 'all_done'
    def __init__(self, pa, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42, render=False, repre='image', end='all_done'):
        super(DeepEnv, self).__init__()
        self.pa = pa
        self.render = render
        # We use image like representation specified in the paper
        self.repre = repre
        # Determine when to terminate,either when there are no new job is there
        # to schedule ('no_new_job') or when all jobs are executed ('all_done')
        self.end = end
        self.nw_dist = pa.dist.bi_model_dist

        self.curr_time = 0

        # set up random seed
        if self.pa.unseen:
            np.random.seed(314159)
        else:
            np.random.seed(seed)

        # Create jobs based on the specified load if
        # no job sequence is input
        if nw_len_seqs is None or nw_size_seqs is None:
            # generate new work
            self.nw_len_seqs, self.nw_size_seqs = \
                self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)
            self.workload = np.zeros(pa.num_res)
            for i in range(pa.num_res):
                self.workload[i] = \
                    np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
                    float(pa.res_slot) / \
                    float(len(self.nw_len_seqs))
            self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                          [self.pa.num_ex, self.pa.simu_len])
            self.nw_size_seqs = np.reshape(self.nw_size_seqs,
                                           [self.pa.num_ex, self.pa.simu_len, self.pa.num_res])
        else:
            # Explicitly specified job sequence
            self.nw_len_seqs = nw_len_seqs
            self.nw_size_seqs = nw_size_seqs

        # The sequence number of the input jobs
        self.seq_no = 0
        # the sequence id of the job in the job sequence
        self.seq_idx = 0

        # Initialize the entire system
        self.machine = Machine(pa)
        self.job_slot = JobSlot(pa)
        self.job_backlog = JobBacklog(pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(pa)

        # Determinitation of the observation space
        state_space = self.observe()
        low_state_space = float(0)
        high_state_space = max(self.machine.colormap)
        self.low_state = low_state_space
        self.high_state = high_state_space
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state,
            shape=(self.pa.network_input_height, self.pa.network_input_width),
            # shape = (20,200)
            dtype=np.int64)

        # Determination of the action space
        # May schedule a job from job_slot or may take void action
        action_space = len(self.job_slot.slot) + 1
        self.state_space = state_space
        self.action_space = spaces.Discrete(action_space)

    def generate_sequence_work(self, simu_len):
        # Function to generate the input job sequence.
        # Here we return the duration of execution of each job (nw_len_seq)
        # and the resource requirement for each job (nw_size_seq).
        # In the resource requirement we specify the number of CPU units
        # and the number of memory units required by the job.

        nw_len_seq = np.zeros(simu_len, dtype=int)
        nw_size_seq = np.zeros((simu_len, self.pa.num_res), dtype=int)

        # Generate a sequence consisting of simu_len jobs
        for i in range(simu_len):
            random_value = np.random.rand()
            if random_value < self.pa.new_job_rate:
                nw_len_seq[i], nw_size_seq[i, :] = self.nw_dist()
        return nw_len_seq, nw_size_seq

    def get_new_job_from_seq(self, seq_no, seq_idx):
        # From the input job sequence, get a job based on the sequence
        # number and sequence id. For e.g if seq_no = 2 and
        # seq_no = 10, then get the 10th job of the 2nd sequence.
        new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx, :],
                      job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.record),
                      enter_time=self.curr_time)
        return new_job

    def observe(self):
        # We return image like representation.
        # The current CPU status of the machine is specified at the beginning
        # followed by the CPU requirements of all the jobs in the job slots.
        # the the current Memory status of the machine is specified along
        # with the memory requirements of all the jobs in the job slots.
        # Finally, at the end we represent the summarized backlog.
        if self.repre == 'image':
            # Calculate the size of the summarized backlog
            backlog_width = int(
                math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
            image_repr = np.zeros(
                (int(self.pa.network_input_height), int(self.pa.network_input_width)))

            ir_pt = 0

            # First all the CPU details (machine+jobs) will be represented,
            # then all the memory details (machine+jobs) will be represented.
            for i in range(self.pa.num_res):

                image_repr[:, ir_pt: ir_pt +
                           self.pa.res_slot] = self.machine.canvas[i, :, :]
                ir_pt += self.pa.res_slot

                for j in range(self.pa.num_nw):

                    if self.job_slot.slot[j] is not None:
                        image_repr[: self.job_slot.slot[j].len,
                                   ir_pt: ir_pt + self.job_slot.slot[j].res_vec[i]] = 1

                    ir_pt += self.pa.max_job_size

            # add the sumarized backlog information
            image_repr[: int(self.job_backlog.curr_size / backlog_width),
                       ir_pt: int(ir_pt + backlog_width)] = 1
            if self.job_backlog.curr_size % backlog_width > 0:
                image_repr[int(self.job_backlog.curr_size / backlog_width),
                           ir_pt: int(ir_pt + self.job_backlog.curr_size % backlog_width)] = 1
            ir_pt += backlog_width

            image_repr[:, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job / \
                float(self.extra_info.max_tracking_time_since_last_job)
            ir_pt += 1

            assert ir_pt == image_repr.shape[1]

            return image_repr

    def plot_state(self):
        # Plot the image representation observation space if we specify
        # render = True
        plt.figure("screen", figsize=(20, 5))

        skip_row = 0

        for i in range(self.pa.num_res):

            plt.subplot(self.pa.num_res,
                        1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                        i * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

            plt.imshow(self.machine.canvas[i, :, :],
                       interpolation='nearest', vmax=1)

            for j in range(self.pa.num_nw):

                job_slot = np.zeros(
                    (self.pa.time_horizon, self.pa.max_job_size))
                if self.job_slot.slot[j] is not None:  # fill in a block of work
                    job_slot[: self.job_slot.slot[j].len,
                             :self.job_slot.slot[j].res_vec[i]] = 1

                plt.subplot(self.pa.num_res,
                            1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                            1 + i * (self.pa.num_nw + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

                plt.imshow(job_slot, interpolation='nearest', vmax=1)

                if j == self.pa.num_nw - 1:
                    skip_row += 1

        skip_row -= 1
        backlog_width = int(
            math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
        backlog = np.zeros((self.pa.time_horizon, backlog_width))

        backlog[: self.job_backlog.curr_size /
                backlog_width, : backlog_width] = 1
        backlog[self.job_backlog.curr_size / backlog_width,
                : self.job_backlog.curr_size % backlog_width] = 1

        plt.subplot(self.pa.num_res,
                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    self.pa.num_nw + 1 + 1)

        plt.imshow(backlog, interpolation='nearest', vmax=1)

        plt.subplot(self.pa.num_res,
                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    self.pa.num_res * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

        extra_info = np.ones((self.pa.time_horizon, 1)) * \
            self.extra_info.time_since_last_new_job / \
            float(self.extra_info.max_tracking_time_since_last_job)

        plt.imshow(extra_info, interpolation='nearest', vmax=1)

        plt.show()

    def get_reward(self):
        # Reward when the objective is to minimize the average job slowdown
        if self.pa.objective_disc == self.pa.objective_slowdown:
            reward = 0
            # Reward based on jobs currently running
            for j in self.machine.running_job:
                reward += self.pa.delay_penalty / float(j.len)
            # Reward based on job in the waiting queue
            for j in self.job_slot.slot:
                if j is not None:
                    reward += self.pa.hold_penalty / float(j.len)
            # Reward based on job in the backlog
            for j in self.job_backlog.backlog:
                if j is not None:
                    reward += self.pa.dismiss_penalty / float(j.len)

        # Reward when the objective is to minimize the average job completing time
        elif self.pa.objective_disc == self.pa.objective_Ctime:
            reward = 0
            remaining_jobs = 0
            for j in self.machine.running_job:
                remaining_jobs += 1
            # Reward based on job in the waiting queue
            for j in self.job_slot.slot:
                if j is not None:
                    remaining_jobs += 1
            # Reward based on job in the backlog
            for j in self.job_backlog.backlog:
                if j is not None:
                    remaining_jobs += 1
            reward = self.pa.delay_penalty * abs(remaining_jobs)

        return reward

    def step(self, a, repeat=True, information=True):

        status = None
        done = False
        reward = 0
        if information == False:
            info = None
        else:
            # Used for slowdown and completion time
            # calculation when training using stable
            # baselines
            info = {}

        # void action
        if a == self.pa.num_nw:
            status = 'MoveOn'
        # implicit void action
        elif self.job_slot.slot[a] is None:
            status = 'MoveOn'
        else:
            # agent takes valid action , we check if
            # allocation possible.
            allocated = self.machine.allocate_job(
                self.job_slot.slot[a], self.curr_time)
            # implicit void action
            if not allocated:
                status = 'MoveOn'
                if information == True:
                    info['Withheld Job'] = self.job_slot.slot[a]
            else:
                status = 'Allocate'

        # Status = 'MoveOn' indicates that the agent
        # does not wish to shedule any further jobs
        # in the current timestep
        if status == 'MoveOn':
            # Increment the current time if status is 'MoveOn'
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)
            self.extra_info.time_proceed()

            # Termination type - end of new job sequence
            if self.end == "no_new_job":
                if self.seq_idx >= self.pa.simu_len:
                    done = True
            # termination type - when all jobs finish execution
            elif self.end == "all_done":
                if self.seq_idx >= self.pa.simu_len and \
                   len(self.machine.running_job) == 0 and \
                   all(s is None for s in self.job_slot.slot) and \
                   all(s is None for s in self.job_backlog.backlog):
                    done = True
                # force termination so that the episode does not run too long
                elif self.curr_time > self.pa.episode_max_length:
                    done = True

            if not done:

                if self.seq_idx < self.pa.simu_len:
                    # Get new job from the input sequence
                    new_job = self.get_new_job_from_seq(
                        self.seq_no, self.seq_idx)
                    self.seq_idx += 1
                    # Check if the job is not null
                    if new_job.len > 0:

                        to_backlog = True
                        for i in range(self.pa.num_nw):
                            # put in the first free available job slot
                            if self.job_slot.slot[i] is None:
                                self.job_slot.slot[i] = new_job
                                self.job_record.record[new_job.id] = new_job
                                to_backlog = False
                                break

                        # If all the job slots are full, the add the job to the backlog
                        if to_backlog:
                            if self.job_backlog.curr_size < self.pa.backlog_size:
                                self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                                self.job_backlog.curr_size += 1
                                self.job_record.record[new_job.id] = new_job
                            else:  # abort, backlog full
                                print("Backlog is full.")
                                # exit(1)

                        self.extra_info.new_job_comes()

            reward = self.get_reward()

        # When status is 'Allocate' assign machine resources to the job. The time will not
        # be increment.
        elif status == 'Allocate':
            # Determine the slowdown and the completion time when the jobs will be
            # assigned computing resources
            if information == True:
                job_slowdown = self.job_slot.slot[a].job_slowdown
                job_completion_time = self.job_slot.slot[a].job_completion_time
                job_length = self.job_slot.slot[a].len
                info['Job Slowdown'] = job_slowdown
                info['Completion Time'] = job_completion_time
                info['Job Length'] = job_length
                info['Allocated Job'] = self.job_slot.slot[a]

            self.job_record.record[self.job_slot.slot[a].id] = self.job_slot.slot[a]
            self.job_slot.slot[a] = None

            # dequeue backlog
            if self.job_backlog.curr_size > 0:
                # if backlog empty, it will be 0
                self.job_slot.slot[a] = self.job_backlog.backlog[0]
                self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -= 1

        ob = self.observe()

        if information == False:
            info = self.job_record

        if done:

            self.seq_idx = 0

            # If the input consists of more than one sequences,
            # start again from the next sequnces and contine
            # till all the sequences are finished
            if repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex
                if self.seq_no != 0:
                    done = False

            # Reset environment
            ob = self.reset()

        if self.render:
            self.plot_state()

        return ob, reward, done, info

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        # Initialize the system
        self.machine = Machine(self.pa)
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)
        obs = self.observe()
        return obs


class Job:
    # The attributes of the job
    def __init__(self, res_vec, job_len, job_id, enter_time):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1
        self.finish_time = -1
        self.job_slowdown = -1
        self.job_completion_time = -1


class JobSlot:
    # The number of job slots in which the jobs will stored
    # and then will be assigned computing resources
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw


class JobBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.record = {}


class Machine:
    # Representation of the machine with two resource types CPU and memory.
    def __init__(self, pa):
        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot

        self.avbl_slot = np.ones(
            (self.time_horizon, self.num_res)) * self.res_slot

        self.running_job = []

        # colormap for graphical representation
        self.colormap = np.arange(
            1 / float(pa.job_num_cap), 1, 1 / float(pa.job_num_cap))
        np.random.shuffle(self.colormap)

        # graphical representation
        self.canvas = np.zeros((pa.num_res, pa.time_horizon, pa.res_slot))

    def allocate_job(self, job, curr_time):
        # Assign the jobs machine resources so that the job can
        # complete its execution

        allocated = False

        for t in range(0, self.time_horizon - job.len):

            new_avbl_res = self.avbl_slot[t: t + job.len, :] - job.res_vec

            # If sufficient resources are available, then schedule the job
            if np.all(new_avbl_res[:] >= 0):

                allocated = True

                self.avbl_slot[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len
                job.job_slowdown = (job.finish_time - job.enter_time) / job.len
                job.job_completion_time = (job.finish_time - job.enter_time)

                # Append the details to a list which store all the
                # currently running jobs
                self.running_job.append(job)

                # update graphical representation used for observation space
                used_color = np.unique(self.canvas[:])
                # Color code with respect to the details in the paper
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                # update the job timings
                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time

                for res in range(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):
                        avbl_slot = np.where(self.canvas[res, i, :] == 0)[0]
                        self.canvas[res, i,
                                    avbl_slot[: job.res_vec[res]]] = new_color

                break

        return allocated

    def time_proceed(self, curr_time):
        # Shift the cluster representation up when the time is incremeted,
        # i.t when the status is 'MoveOn'
        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot

        slowdown = []
        ctime = []
        processed_jobs = []
        for job in self.running_job:
            if job.finish_time <= curr_time:
                slowdown.append(job.job_slowdown)
                ctime.append(job.job_completion_time)
                processed_jobs.append(job)

        self.running_job = [
            job for job in self.running_job if job not in processed_jobs]

        # update graphical representation
        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0


class ExtraInfo:
    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1
