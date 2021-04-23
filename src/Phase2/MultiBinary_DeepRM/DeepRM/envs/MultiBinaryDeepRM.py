import numpy as np
import math
import matplotlib.pyplot as plt
import theano
import other_agents
import parameters
import gym
from gym import spaces
import os


class Env(gym.Env):
    def __init__(self, pa, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42, render=False, repre='image', end='all_done'):
        super(Env, self).__init__()
        self.pa = pa
        self.render = render
        self.repre = repre  # image or compact representation
        self.end = end  # termination type, 'no_new_job' or 'all_done'

        self.nw_dist = pa.dist.bi_model_dist

        self.curr_time = 0

        # set up random seed
        if self.pa.unseen:
            np.random.seed(314159)
        else:
            np.random.seed(seed)

        if nw_len_seqs is None or nw_size_seqs is None:
            # generate new work
            self.nw_len_seqs, self.nw_size_seqs = \
                self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)

            self.workload = np.zeros(pa.num_res)
            # cluser load calculation
            for i in range(pa.num_res):
                self.workload[i] = \
                    np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
                    float(pa.res_slot) / \
                    float(len(self.nw_len_seqs))
                # print("Load on # " + str(i) +
                #       " resource dimension is " + str(self.workload[i]))
            self.pa.cluster_load = np.sum(self.workload)
            print("Cluster Load is :", self.pa.cluster_load)
            self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                          [self.pa.num_ex, self.pa.simu_len])
            self.nw_size_seqs = np.reshape(self.nw_size_seqs,
                                           [self.pa.num_ex, self.pa.simu_len, self.pa.num_res])
        else:
            self.nw_len_seqs = nw_len_seqs
            self.nw_size_seqs = nw_size_seqs

        self.seq_no = 0  # which example sequence
        self.seq_idx = 0  # index in that sequence

        # initialize system
        self.machine = Machine(pa)
        self.job_slot = JobSlot(pa)
        self.job_backlog = JobBacklog(pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(pa)
        state_space = self.observe()
        low_state_space = float(0)
        high_state_space = max(self.machine.colormap)
        self.low_state = low_state_space
        self.high_state = high_state_space

        # may schedule a job from job_slot or may take null acction
        action_space = len(self.job_slot.slot)
        self.state_space = state_space
        self.action_space = spaces.MultiBinary(action_space)
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, shape=(self.pa.network_input_height, self.pa.network_input_width), dtype=np.int64)

    def generate_sequence_work(self, simu_len):

        nw_len_seq = np.zeros(simu_len, dtype=int)
        nw_size_seq = np.zeros((simu_len, self.pa.num_res), dtype=int)

        for i in range(simu_len):
            random_value = np.random.rand()
            if random_value < self.pa.new_job_rate:  # a new job comes

                nw_len_seq[i], nw_size_seq[i, :] = self.nw_dist()

        return nw_len_seq, nw_size_seq

    def get_new_job_from_seq(self, seq_no, seq_idx):
        new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx, :],
                      job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.record),
                      enter_time=self.curr_time)
        return new_job

    def observe(self):
        if self.repre == 'image':

            backlog_width = int(
                math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))

            image_repr = np.zeros(
                (int(self.pa.network_input_height), int(self.pa.network_input_width)))

            ir_pt = 0

            for i in range(self.pa.num_res):

                image_repr[:, ir_pt: ir_pt +
                           self.pa.res_slot] = self.machine.canvas[i, :, :]
                ir_pt += self.pa.res_slot

                for j in range(self.pa.num_nw):

                    if self.job_slot.slot[j] is not None:  # fill in a block of work
                        image_repr[: self.job_slot.slot[j].len,
                                   ir_pt: ir_pt + self.job_slot.slot[j].res_vec[i]] = 1

                    ir_pt += self.pa.max_job_size

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

        elif self.repre == 'compact':

            compact_repr = np.zeros(self.pa.time_horizon * (self.pa.num_res + 1) +  # current work
                                    # new work
                                    self.pa.num_nw * (self.pa.num_res + 1) +
                                    1)

            cr_pt = 0

            # current work reward, after each time step, how many jobs left in the machine
            job_allocated = np.ones(self.pa.time_horizon) * \
                len(self.machine.running_job)
            for j in self.machine.running_job:
                job_allocated[j.finish_time - self.curr_time:] -= 1

            compact_repr[cr_pt: cr_pt + self.pa.time_horizon] = job_allocated
            cr_pt += self.pa.time_horizon

            # current work available slots
            for i in range(self.pa.num_res):
                compact_repr[cr_pt: cr_pt +
                             self.pa.time_horizon] = self.machine.avbl_slot[:, i]
                cr_pt += self.pa.time_horizon

            # new work duration and size
            for i in range(self.pa.num_nw):

                if self.job_slot.slot[i] is None:
                    compact_repr[cr_pt: cr_pt + self.pa.num_res + 1] = 0
                    cr_pt += self.pa.num_res + 1
                else:
                    compact_repr[cr_pt] = self.job_slot.slot[i].len
                    cr_pt += 1

                    for j in range(self.pa.num_res):
                        compact_repr[cr_pt] = self.job_slot.slot[i].res_vec[j]
                        cr_pt += 1

            # backlog queue
            compact_repr[cr_pt] = self.job_backlog.curr_size
            cr_pt += 1

            # fill up the compact representation vector
            assert cr_pt == len(compact_repr)

            return compact_repr

    def plot_state(self):
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

        plt.show()     # manual
        # plt.pause(0.01)  # automatic

    def get_reward(self):
        reward = 0
        if self.pa.objective != self.pa.A2C_Ctime and self.pa.objective != self.pa.PPO2_Ctime and self.pa.objective != self.pa.ACKTR_Ctime:
            # Reward based on jobs currently running
            for j in self.machine.running_job:
                reward += self.pa.penalty / float(j.len)
            # Reward based on job in the waiting queue
            for j in self.job_slot.slot:
                if j is not None:
                    reward += self.pa.penalty / float(j.len)
            # Reward based on job in the backlog
            for j in self.job_backlog.backlog:
                if j is not None:
                    reward += self.pa.penalty / float(j.len)

        elif self.pa.objective == self.pa.A2C_Ctime or self.pa.objective == self.pa.PPO2_Ctime or self.pa.objective == self.pa.PPO2_Ctime:
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
            reward = self.pa.penalty * abs(remaining_jobs)

        return reward

    def step(self, act, repeat=True, information=True):
        status = None
        final_status = 'MoveOn'
        moveOncount = 0
        allocationcount = 0
        done = False
        reward = 0
        if information == False:
            info = None
        else:
            info = {}
            info['Withheld Job'] = []
            info['Allocated Job'] = []

        for a in range(len(act)):
            if act[a] == 1 or 1 not in act: 
                if 1 not in act:  # explicit void action
                    status = 'MoveOn'
                elif self.job_slot.slot[a] is None:  # implicit void action
                    status = 'MoveOn'
                else:
                    allocated = self.machine.allocate_job(
                        self.job_slot.slot[a], self.curr_time)
                    if not allocated:  # implicit void action
                        status = 'MoveOn'
                        if self.job_slot.slot[a] != None and self.job_slot.slot[a] not in info['Withheld Job']:
                            info['Withheld Job'].append(self.job_slot.slot[a])
        
                    else:
                        status = 'Allocate'
                        final_status = 'Allocate'

                if status == 'Allocate':
                    allocationcount += 1
                    self.job_record.record[self.job_slot.slot[a].id] = self.job_slot.slot[a]
                    info['Allocated Job'].append(self.job_slot.slot[a])
                    self.job_slot.slot[a] = None

                    # dequeue backlog
                    if self.job_backlog.curr_size > 0:
                        # if backlog empty, it will be 0
                        self.job_slot.slot[a] = self.job_backlog.backlog[0]
                        self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
                        self.job_backlog.backlog[-1] = None
                        self.job_backlog.curr_size -= 1

                if status == 'MoveOn':  
                    moveOncount += 1
                    self.curr_time += 1
                    self.machine.time_proceed(self.curr_time, info)
                    self.extra_info.time_proceed()
                    
                    if self.end == "no_new_job":  # end of new job sequence
                        if self.seq_idx >= self.pa.simu_len:
                            done = True
                    elif self.end == "all_done":  # everything has to be finished
                        if self.seq_idx >= self.pa.simu_len and \
                        len(self.machine.running_job) == 0 and \
                        all(s is None for s in self.job_slot.slot) and \
                        all(s is None for s in self.job_backlog.backlog):
                            done = True
                        elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                            done = True

                    if not done:
                        if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
                            new_job = self.get_new_job_from_seq(
                                self.seq_no, self.seq_idx)
                            self.seq_idx += 1
                            if new_job.len > 0:  # a new job comes

                                to_backlog = True
                                for i in range(self.pa.num_nw):
                                    # put in new visible job slots
                                    if self.job_slot.slot[i] is None:
                                        self.job_slot.slot[i] = new_job
                                        self.job_record.record[new_job.id] = new_job
                                        to_backlog = False
                                        break

                                if to_backlog:
                                    if self.job_backlog.curr_size < self.pa.backlog_size:
                                        self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                                        self.job_backlog.curr_size += 1
                                        self.job_record.record[new_job.id] = new_job
                                    else:  # abort, backlog full
                                        print("Backlog is full.")
                                        # exit(1)

                                self.extra_info.new_job_comes()
                                
                    if moveOncount == self.pa.move_on_count:
                       break
        
        if not allocationcount > 1:
            reward = self.get_reward()

        ob = self.observe()
        if information == False:
            info = self.job_record

        if done:
            # print("The current time" , self.curr_time)
            self.seq_idx = 0

            if repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex
                if self.seq_no != 0:
                    done = False

            ob = self.reset()

        if self.render:
            self.plot_state()

        return ob, reward, done, info

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        # initialize system
        self.machine = Machine(self.pa)
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)
        obs = self.observe()
        return obs


class Job:
    def __init__(self, res_vec, job_len, job_id, enter_time):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1
        self.job_completion_time = -1
        self.job_slowdown = -1


class JobSlot:
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

        allocated = False

        for t in range(0, self.time_horizon - job.len):

            new_avbl_res = self.avbl_slot[t: t + job.len, :] - job.res_vec

            if np.all(new_avbl_res[:] >= 0):

                allocated = True

                self.avbl_slot[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len
                job.job_completion_time = float(
                    job.finish_time - job.enter_time)
                job.job_slowdown = (job.finish_time - job.enter_time) / job.len
                self.running_job.append(job)

                # update graphical representation

                used_color = np.unique(self.canvas[:])
                # WARNING: there should be enough colors in the color map
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

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

    def time_proceed(self, curr_time, info):
        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot
        processed_jobs = []
        for job in self.running_job :
            if job.finish_time <= curr_time :
                job.finish_time = curr_time
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


# ==========================================================================
# ------------------------------- Unit Tests -------------------------------
# ==========================================================================


# def test_backlog():
#     print("Start Test Backlog")
#     pa = parameters.Parameters()
#     pa.num_nw = 5
#     pa.simu_len = 50
#     pa.num_ex = 10
#     pa.new_job_rate = 1
#     print("The job rate is 1 ", pa.new_job_rate)
#     pa.compute_dependent_parameters()

#     env = Env(pa, render=False, repre='image')

#     env.step(5)
#     env.step(5)
#     env.step(5)
#     env.step(5)
#     env.step(5)

#     env.step(5)
#     assert env.job_backlog.backlog[0] is not None
#     assert env.job_backlog.backlog[1] is None
#     print("New job is backlogged.")

#     env.step(5)
#     env.step(5)
#     env.step(5)
#     env.step(5)

#     job = env.job_backlog.backlog[0]
#     env.step(0)
#     assert env.job_slot.slot[0] == job

#     job = env.job_backlog.backlog[0]
#     env.step(0)
#     assert env.job_slot.slot[0] == job

#     job = env.job_backlog.backlog[0]
#     env.step(1)
#     assert env.job_slot.slot[1] == job

#     job = env.job_backlog.backlog[0]
#     env.step(1)
#     assert env.job_slot.slot[1] == job

#     env.step(5)

#     job = env.job_backlog.backlog[0]
#     env.step(3)
#     assert env.job_slot.slot[3] == job

#     print("- Backlog test passed -")


# def test_compact_speed():
#     print("Start Test Compact Speed")
#     pa = parameters.Parameters()
#     pa.simu_len = 100
#     pa.num_ex = 1
#     pa.new_job_rate = 0.635
#     pa.compute_dependent_parameters()
#     print("The job rate is 0.635", pa.new_job_rate)
#     env = Env(pa, render=False, repre='compact')

#     import other_agents
#     import time

#     start_time = time.time()
#     for i in range(100000):
#         a = other_agents.get_sjf_action(env.machine, env.job_slot)
#         env.step(a)
#     end_time = time.time()
#     print("- Elapsed time: ", end_time - start_time, "sec -")


# def test_image_speed():
#     print("Start Test Image Speed")
#     pa = parameters.Parameters()
#     # pa.simu_len = 50
#     # pa.num_ex = 10
#     # pa.new_job_rate = 0.7
#     pa.simu_len = 60
#     pa.num_ex = 1
#     pa.num_nw = 10
#     pa.num_seq_per_batch = 20
#     pa.new_job_rate = 0.6
#     print("The job rate is 0.7", pa.new_job_rate)
#     pa.compute_dependent_parameters()

#     env = Env(pa, render=False, repre='image')

#     import other_agents
#     import time

#     start_time = time.time()
#     for i in range(100000):
#         a = other_agents.get_sjf_action(env.machine, env.job_slot)
#         env.step(a)
#     end_time = time.time()
#     print("- Elapsed time: ", end_time - start_time, "sec -")


# if __name__ == '__main__':
#     test_backlog()
#     test_compact_speed()
#     test_image_speed()

    