import numpy as np


def agent(job_slot):
    num_act = len(job_slot.slot) + 1
    # if no action available,
    action = np.random.randint(num_act)
    return action
