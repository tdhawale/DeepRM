import pandas as pd
import numpy as np
from IPython.display import display

def generate_sequence_work(n,m):

    print("value of n",n)
    print("value of m",m)

    dataframe = pd.read_csv("/home/aicon/kunalS/workspace/csvTest/container_usage.csv")
    # print(dataframe.head(10))

    np_array = dataframe.to_numpy()
    print("************")
    # display(np_array)

    job_sequence_len = []
    job_sequence_size = []

    subArray = np_array[n:m]
    print("length of the array is -> ", len(subArray))
    # display(subArray)
    # print("subArray -> ", subArray)

    for i in range(len(subArray)):
        job_sequence_len.append(int(subArray[i][0] / 10000))
        
        if(subArray[i][2] > 10):
            job_sequence_size.append([int(subArray[i][2]/10), int(subArray[i][3] / 10)])
        else:
            job_sequence_size.append([int(subArray[i][2]), int(subArray[i][3] / 10)])

    # print("job_sequence_len -> ", len(job_sequence_len))
    # print("job_sequence_len values -> ", job_sequence_len)

    # print("job_sequence_size -> ", len(job_sequence_size))
    # print("job_sequence_size values -> ", job_sequence_size)

# slice_1_a = subArray[:, 0]
# print("length of the slice_1_a is -> ",len(slice_1_a))
# # display(slice_1)
# print("slice_1_a",slice_1_a)

# slice_2_b = subArray[:, 2:4]
# print("length of the slice_2_b is -> ",len(slice_2_b))
# # display(slice_1)
# print("slice_2_b",slice_2_b)

    return job_sequence_len, job_sequence_size
