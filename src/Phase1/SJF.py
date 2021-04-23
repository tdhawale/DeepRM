import numpy as np
import parameters


def agent(machine, job_slot):
    sjf_value = 10000
    pa = parameters.Parameters()
    action = len(job_slot.slot)  # Void action = 5

    for i in range(len(job_slot.slot)):
        job = job_slot.slot[i]
        if job is not None:
            for t in range(0, pa.time_horizon - job.len):
                # Check available resoure.
                available_res = machine.available_res_slot[t:t+job.len, :]
                resource_left = available_res - job.resource_requirement
                # If enough resource is avaiable only then take action. Or else void action will be performed.
                # enough resource to allocate
                if np.all(resource_left[:] >= 0):
                    # Action with the shortest job length will be stored temporarily
                    temp_sjf_value = job.len
                    if temp_sjf_value < sjf_value:
                        sjf_value = temp_sjf_value
                        action = i
    return action
