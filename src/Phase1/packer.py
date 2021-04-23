import numpy as np
import parameters


def agent(machine, job_slot):
    value = 0
    pa = parameters.Parameters()
    action = len(job_slot.slot)  # if no action available, hold

    for i in range(len(job_slot.slot)):
        job = job_slot.slot[i]
        if job is not None:
            # checking the available resource
            for t in range(0, pa.time_horizon - job.len):
                available_res = machine.available_res_slot[t:t+job.len, :]
                resource_left = available_res - job.resource_requirement

                # enough resource to allocate
                if np.all(resource_left[:] >= 0):
                    temp_value = available_res[0, :].dot(
                        job.resource_requirement)
                    if temp_value > value:
                        value = temp_value
                        action = i

    return action
