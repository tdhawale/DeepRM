# Scheduling

**Advisor: Asif Hasnain**

**Members**

* Aluri Mohini
* Kunal Pratap Singh Sisodia
* Tejas Ravindra Dhawale

## Details :

The code base is divided into two phases. 
* Phase 1.
* Phase 2.

## Phase 1:

* It contains the code for the first phase of implementation. The implementation was based on the stacked 
representation of the observation space. Also, the observation space did not contain the backlog component. For detailed overview of observation space refer [report](https://github.com/CN-UPB/pg-aicon/blob/master/Scheduling/documentation/Employing_Reinforcement_Learning_Algorithms_to_solve_Cluster_Scheduling_Problem.pdf). 
* Custom logic for determination of cluster load.
* Execution steps are specified in [Phase 1](https://github.com/CN-UPB/pg-aicon/blob/master/Scheduling/src/Phase1).


## Phase 2:

* It contains the code for the second phase of implementation. The implementation was based on the image like 
representation of the observation space. Summarized backlog information also added in the observation space. For detailed overview of observation space refer [report](https://github.com/CN-UPB/pg-aicon/blob/master/Scheduling/documentation/Employing_Reinforcement_Learning_Algorithms_to_solve_Cluster_Scheduling_Problem.pdf).
* Redesigned logic for cluster load calculation.
* Execution steps are specified in [Phase 2](https://github.com/CN-UPB/pg-aicon/blob/master/Scheduling/src/Phase2).


