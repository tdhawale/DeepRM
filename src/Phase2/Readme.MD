# How to execute the project DeepRL.

This readme will help helps an ordinary user (non developer) to execute and get the results with a single command.

# Makefile commands

## Environment Setup 

### `make application_update`

This will update the list of prerequisite packages.

### `make install_miniconda`

This will download and install miniconda 3.

### `make upgrade_miniconda`

Restart your shell and run this command to upgrade all prerequisties for miniconda 3.

### `make pip_files`

This will install all the basic services like gym and the required environments.

### `make lasagne_theano`

This will install Lasagne and theano packages.

### `make raytune`

This will install raytune and required prerequisties.

### `make clean`

This will help clean the cache in the project.
After completing the pre-setup using the above commands, you can execute the following commands to get the results of the discrete action space.


## Steps to perform for Discrete Environment
### `make run-train`

This command will train all the agents.

### `make run-load_Variation_json`

This command will generate the results for [DeepRM](https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf) .

### `make run-load_Variation_Discrete`

This command will generate the results for stable Baseline trained agents for varying load.

### `make run-run`

This command will generate the performance results.

### `make run-Withheld_Jobs_Discrete`

This command will generate the graph for Withheld Jobs ( performacne gain of trained agents is because of the fact that they are able to hold larger jobs even if resources are available).

### `make run-job_generation`

This command will generate graph to help you understand how workload is calculated and how the jobs are generated.

### `make run-raytune_discrete`

This command will tune the hyperparameters of the specified models.

## Steps to perform for Multi-Binary Environment
### `make run-Train_all`

This command will train all the agents.

### `make run-LoadVariation`

This command will generate the results for stable Baseline trained agents for varying load.

### `make run-Script`

This command will generate the performance results.

### `make run-Withhold_Jobs`

This command will generate the graph for Withheld Jobs ( performance gain of trained agents is because of the fact that they are able to hold larger jobs even if resources are available).
