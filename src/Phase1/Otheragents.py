import numpy as np
import random
import parameters


def get_sjf_action(machine, job_slot):
    act = len(job_slot.slot)  # if no action available, hold
    len_list = []
    K = 1
    for i in range(len(job_slot.slot)):
        new_job = job_slot.slot[i]
        if new_job is not None:  # there is a pending job
            tmp_sjf_score = 1 / float(new_job.len)
            len_list.append(tmp_sjf_score)
        else:
            len_list.append(0)

    # k shortest jobs. Can be less than k if there are None job entries
    res = sorted(range(len(len_list)), reverse=True,
                 key=lambda sub: len_list[sub])[:K]

    for i in range(len(res)):
        index = res[i]
        new_job = job_slot.slot[index]
        if len_list[index] != 0:  # there is a pending job
            avbl_res = machine.available_res_slot[new_job.len, :]
            res_left = avbl_res - new_job.resource_requirement
            if np.all(res_left[:] >= 0):  # enough resource to allocate
                act = index

    return act


def get_packer_action(machine, job_slot):
    act = len(job_slot.slot)  # if no action available, hold
    res_req_list = []
    K = 1
    for i in range(len(job_slot.slot)):
        new_job = job_slot.slot[i]
        avbl_res = machine.available_res_slot
        if new_job is not None:  # there is a pending job
            tmp_align_score = avbl_res[0, :].dot(new_job.resource_requirement)
            res_req_list.append(tmp_align_score)
        else:
            res_req_list.append(0)

    # k jobs having higher resorce_requirement. Can be less than k if there are None job entries
    res = sorted(range(len(res_req_list)), reverse=True,
                 key=lambda sub: res_req_list[sub])[:K]
    for i in range(len(res)):
        index = res[i]
        new_job = job_slot.slot[index]
        if res_req_list[index] != 0:  # there is a pending job
            avbl_res = machine.available_res_slot[new_job.len, :]
            res_left = avbl_res - new_job.resource_requirement
            if np.all(res_left[:] >= 0):  # enough resource to allocate
                act = index

    return act


def rand_key(machine, job_slot):
    act = []  # if no action available, hold
    K = 1
    action = len(job_slot.slot)
    for i in range(len(job_slot.slot)):
        act.append(0)
        bin = random.randint(0, 1)
        if bin == 1 and act.count(1) < K:
            new_job = job_slot.slot[i]
            if new_job is not None:
                avbl_res = machine.available_res_slot[new_job.len, :]
                res_left = avbl_res - new_job.resource_requirement
                if np.all(res_left[:] >= 0):  # enough resource to allocate
                    act[i] = random.randint(0, 1)
                    if act[i] == 1:
                        action = i

    return action
