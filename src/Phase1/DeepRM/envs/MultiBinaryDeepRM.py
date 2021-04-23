import numpy as np
import os
import gym
from gym import spaces


class Env(gym.Env):
    # We pass a dictionary of parameters along with job requirements and also we set
    # the end type to 'all_done'
    def __init__(self, pa, job_sequence_len, job_sequence_size, end='all_done'):
        super(Env, self).__init__()
        self.pa = pa
        self.end = end  # termination type, 'no_new_job' or 'all_done'
        self.curr_time = 0
        self.job_sequence_len = job_sequence_len
        self.job_sequence_size = job_sequence_size
        self.seq_no = 0  # which example sequence
        self.seq_idx = 0  # index in that sequence

        # initialize system
        self.machine = Machine(pa)
        self.job_slot = JobSlot(pa)
        self.job_backlog = JobBacklog(pa)
        self.job_record = JobRecord()
        state_space = self.observe()
        low_state_space = []
        high_state_space = []
        # define the lowest and highest possible state values from default state space
        for i in range(len(state_space)):
            high_state_space.append(state_space[0])
            low_state_space.append(state_space[1])
        self.low_state = np.array(low_state_space)
        self.high_state = np.array(high_state_space)

        # may schedule a job from job_slot or may take null acction
        action_space = len(self.job_slot.slot)
        self.state_space = state_space
        self.action_space = spaces.MultiBinary(action_space)
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.int64)

    def step(self, a):
        status = None
        final_status = 'MoveOn'
        done = False
        info = {}
        info['Withheld Job'] = []
        info['Allocated Job'] = []
        # iterate over valid actions
        for act in range(len(a)):
            if (a[act] == 1) or (1 not in a):
                if 1 not in a or act == self.pa.job_wait_queue or self.job_slot.slot[act] is None:
                    status = 'MoveOn'
                else:
                    allocated = self.machine.allocate_job(
                        self.job_slot.slot[act], self.curr_time)
                    if not allocated:  # implicit void action
                        status = 'MoveOn'
                        if self.job_slot.slot[act] != None and self.job_slot.slot[act] not in info['Withheld Job']:
                            info['Withheld Job'].append(
                                self.job_slot.slot[act])
                    else:
                        status = 'Allocate'
                        final_status = 'Allocate'

                if status == 'Allocate':
                    info['Allocated Job'].append(self.job_slot.slot[act])

                    self.job_record.record[self.job_slot.slot[act].id] = self.job_slot.slot[act]
                    self.job_slot.slot[act] = None

                    # dequeue backlog
                    if self.job_backlog.curr_size > 0:
                        # if backlog empty, it will be 0
                        self.job_slot.slot[act] = self.job_backlog.backlog[0]
                        self.job_backlog.backlog[: -
                                                 1] = self.job_backlog.backlog[1:]
                        self.job_backlog.backlog[-1] = None
                        self.job_backlog.curr_size -= 1

        if final_status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)

            if self.end == "no_new_job":  # end of new job sequence
                if self.seq_idx >= self.pa.simu_len:
                    done = True
            elif self.end == "all_done":
                # Done is true if no jobs are running in th machine + no job in the waiting queue and
                # no jobs in the backlog
                if self.seq_idx >= self.pa.simu_len and len(self.machine.running_job) == 0 and \
                        all(s is None for s in self.job_slot.slot) and \
                        all(s is None for s in self.job_backlog.backlog):
                    done = True

            if not done:
                if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
                    new_job = self.get_new_job_from_seq(
                        self.seq_no, self.seq_idx)
                    self.seq_idx += 1
                    if new_job.len > 0:
                        add_to_backlog = True
                        # If the job slot is empty then only put the new job in the job slot/waiting queue
                        # If the job slot is not empty then the job will go in the job backlog
                        for i in range(self.pa.job_wait_queue):
                            # put in new visible job slots
                            if self.job_slot.slot[i] is None:
                                self.job_slot.slot[i] = new_job
                                self.job_record.record[new_job.id] = new_job
                                add_to_backlog = False
                                break

                        if add_to_backlog:
                            if self.job_backlog.curr_size < self.pa.backlog_size:
                                self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                                self.job_backlog.curr_size += 1
                                self.job_record.record[new_job.id] = new_job
                            else:  # abort, backlog full
                                print("Backlog full.")

        reward = self.get_reward()
        ob = self.observe()
        if done:
            self.seq_idx = 0
            self.seq_no = 0
            self.reset()
        return ob, reward, done, info

    # Get new job from waiting queue
    def get_new_job_from_seq(self, seq_no, seq_idx):
        new_job = Job(resource_requirement=self.job_sequence_size[seq_idx], job_len=self.job_sequence_len[seq_idx],
                      job_id=len(self.job_record.record), enter_time=self.curr_time)
        return new_job

    def observe(self):
        ob = []
        ob.append(self.machine.available_res_slot)

        # The resource profile of jobs in the job slot queue/waiting queue/M
        for i in self.job_slot.slot:
            resource_profile = np.zeros(
                (self.pa.time_horizon, self.pa.num_resources))
            if i is not None:
                for len in range(i.len):
                    resource_profile[len] = i.resource_requirement

            ob.append(resource_profile)

        for i in self.job_backlog.backlog:
            resource_profile = np.zeros(
                (self.pa.time_horizon, self.pa.num_resources))
            if i is not None:
                for len in range(i.len):
                    resource_profile[len] = i.resource_requirement

            ob.append(resource_profile)

        return np.array(ob)

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0
        # initialize system
        self.machine = Machine(self.pa)
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.job_record = JobRecord()
        obs = self.observe()
        return obs

    def get_reward(self):
        reward = 0
        if self.pa.objective != self.pa.A2C_Ctime:
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

        elif self.pa.objective == self.pa.A2C_Ctime:
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


class Job:
    def __init__(self, resource_requirement, job_len, job_id, enter_time):
        self.id = job_id
        self.resource_requirement = resource_requirement
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1
        self.job_completion_time = -1
        self.job_slowdown = -1


class JobSlot:
    def __init__(self, pa):
        self.slot = [None] * pa.job_wait_queue


class JobBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.record = {}


class Machine:
    def __init__(self, pa):
        self.num_resources = pa.num_resources
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot
        self.available_res_slot = np.ones(
            (self.time_horizon, self.num_resources)) * self.res_slot
        self.running_job = []

    def allocate_job(self, job, curr_time):
        allocated = False
        for t in range(0, self.time_horizon - job.len):
            new_avbl_res = self.available_res_slot[t: t +
                                                   job.len, :] - job.resource_requirement
            if np.all(new_avbl_res[:] >= 0):
                allocated = True
                self.available_res_slot[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len
                job.job_completion_time = float(
                    job.finish_time - job.enter_time)
                job.job_slowdown = (job.finish_time - job.enter_time) / job.len
                self.running_job.append(job)
                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                break
        return allocated

    def time_proceed(self, curr_time):
        self.available_res_slot[:-1, :] = self.available_res_slot[1:, :]
        self.available_res_slot[-1, :] = self.res_slot
        for job in self.running_job:
            if job.finish_time <= curr_time:
                self.running_job.remove(job)
