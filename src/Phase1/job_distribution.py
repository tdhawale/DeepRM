import numpy as np
import random

class Dist:

    def __init__(self, num_resources, max_nw_size, job_len,job_small_chance):
        self.num_resources = num_resources
        self.max_nw_size = max_nw_size
        self.job_len = job_len

        self.job_small_chance = job_small_chance

        self.job_len_big_lower = job_len * 2 / 3
        self.job_len_big_upper = job_len

        self.job_len_small_lower = 1
        self.job_len_small_upper = job_len / 5

        self.dominant_res_lower = 0.25 * max_nw_size
        self.dominant_res_upper = 0.5 * max_nw_size

        self.other_res_lower = max(0.05 * max_nw_size, 1)
        self.other_res_upper = 0.1 * max_nw_size

    def normal_dist(self):

        # new work duration
        nw_len = np.random.randint(1, self.job_len + 1)  # same length in every dimension

        nw_size = np.zeros(self.num_resources)

        for i in range(self.num_resources):
            nw_size[i] = np.random.randint(1, self.max_nw_size + 1)

        return nw_len, nw_size

    def bi_model_dist(self,pa):

        # -- job length --
        if np.random.rand() < self.job_small_chance:  # small job
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)
        else:  # big job
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)

        nw_size = np.zeros(self.num_resources)

        # -- job resource request --
        dominant_res = pa.dominant_res # 0 and 1 for respecive indixes and 2 for both resources
        if dominant_res == 3:
            dominant_res = random.randint(0, 1)
            
        for i in range(self.num_resources):
            if i == dominant_res or dominant_res == 2:
                nw_size[i] = np.random.randint(self.dominant_res_lower,
                                               self.dominant_res_upper + 1)
            else:
                nw_size[i] = np.random.randint(self.other_res_lower,
                                               self.other_res_upper + 1)
        return nw_len, nw_size


def generate_sequence_work(pa):

    np.random.seed(pa.random_seed)

    simu_len = pa.simu_len * pa.num_ex

    nw_dist = pa.dist.bi_model_dist

    nw_len_seq = np.zeros(simu_len, dtype=int)
    nw_size_seq = np.zeros((simu_len, pa.num_resources), dtype=int)
    cluster_occupied = 0
    
    for i in range(simu_len):

        if np.random.rand() < pa.new_job_rate and cluster_occupied <= (pa.cluster_capacity * pa.cluster_load):  # a new job comes

            nw_len_seq[i], nw_size_seq[i, :] = nw_dist(pa)
            cluster_occupied = cluster_occupied + (nw_len_seq[i] * nw_size_seq[i][0] * nw_size_seq[i][1])

    pa.cluster_occupied = cluster_occupied  
    nw_size_seq_list = []
    for i in range(len(nw_size_seq)):
        nw_size_seq_list.append(list(nw_size_seq[i]))

    return list(nw_len_seq), nw_size_seq_list

# compute simulen and arrival_rate wrt max cluser load passed
def compute_simulen_and_arrival_rate(max_load,pa):
    max_load = max_load
    pa.new_job_rate = 1
    loads = [max_load, pa.cluster_load]
    np.random.seed(pa.random_seed)
    random.seed(pa.random_seed)
    nw_dist = pa.dist.bi_model_dist
    nw_len_seq = []
    nw_size_seq = []
    req_len_seq = []
    req_size_seq = []
    cluster_capacity = pa.cluster_capacity
    
    for load in loads:
        cluster_occupied = 0
        while cluster_occupied <= (cluster_capacity * load):
            len_i = 0
            size_i = [0, 0]
            if np.random.rand() < pa.new_job_rate:  # a new job comes
                len_i, size_i = nw_dist(pa)
                cluster_occupied = cluster_occupied + (len_i * size_i[0] * size_i[1])

            if load == max_load:
                nw_len_seq.append(len_i)
                nw_size_seq.append(size_i)
            elif load == pa.cluster_load:
                req_len_seq.append(len_i)
                req_size_seq.append(size_i)

        if max_load == pa.cluster_load:
            req_len_seq = nw_len_seq
            break
        
    simu_len = len(nw_len_seq)
    new_job_arrival_rate = len(req_len_seq) / len(nw_len_seq)
    return simu_len, new_job_arrival_rate