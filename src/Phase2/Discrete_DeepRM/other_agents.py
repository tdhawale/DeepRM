import numpy as np

# Packer Algorithm 

def get_packer_action(machine, job_slot):
        align_score = 0
        act = len(job_slot.slot)  
        # Hold the job if no action can be taken 

        for i in range(len(job_slot.slot)):
            new_job = job_slot.slot[i]
            if new_job is not None:  
                # If there is a pending job 

                avbl_res = machine.avbl_slot[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  
                    # if there is sufficient resource to be allocated to the job

                    tmp_align_score = avbl_res[0, :].dot(new_job.res_vec)

                    if tmp_align_score > align_score:
                        align_score = tmp_align_score
                        act = i
        return act
        # returns the action of the job which is the largest in the given job slot.

# Shortest Job first algorithm 

def get_sjf_action(machine, job_slot):
        sjf_score = 0
        act = len(job_slot.slot)  

        for i in range(len(job_slot.slot)):
            new_job = job_slot.slot[i]
            if new_job is not None:

                avbl_res = machine.avbl_slot[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  
                    # Checkin for the sufficient resources.

                    tmp_sjf_score = 1 / float(new_job.len)

                    if tmp_sjf_score > sjf_score:
                        sjf_score = tmp_sjf_score
                        act = i
        return act
        # returns the action for the smallest job from the job slot

# 
def get_packer_sjf_action(machine, job_slot, knob):  
    # knob is the flag which helps to take actions like packer if it is 1 and actions will be taken like sjf if knob is 0.
    
        combined_score = 0
        act = len(job_slot.slot) 

        for i in range(len(job_slot.slot)):
            new_job = job_slot.slot[i]
            if new_job is not None:  

                avbl_res = machine.avbl_slot[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  

                    tmp_align_score = avbl_res[0, :].dot(new_job.res_vec)
                    tmp_sjf_score = 1 / float(new_job.len)

                    tmp_combined_score = knob * tmp_align_score + (1 - knob) * tmp_sjf_score

                    if tmp_combined_score > combined_score:
                        combined_score = tmp_combined_score
                        act = i
        return act
        # returns an action with repect to knob if 1 then it works like packer and 0 then SJF.

# Random action alorithm
def get_random_action(job_slot):
    num_act = len(job_slot.slot) + 1  # if no action available,
    act = np.random.randint(num_act)
    return act
    # returns a random number for action using the random library.
