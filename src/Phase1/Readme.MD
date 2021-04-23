# How to execute the project DeepRL.

This readme will help helps an ordinary user (non developer) to execute and get the results with a single command.

# Makefile commands

### `make install_miniconda`

This will download and install miniconda 3.

### `make upgrade_miniconda`

Restart your shell and run this command to upgrade all prerequisties for miniconda 3.

### `make pip_files`

This will install all the basic services like gym and the required environments.

### `make raytune`

This will install raytune and required prerequisties.

### `make clean`

This will help clean the cache in the project.


After completing the pre-setup using the above commands, you can execute the following commands to train discrete environment agents.

### `make run-stableBaseLine_Models_Training`

This command will run the training for the DeepRl agents.


On the other hand, once the pre-setup is complete, the following commands can be executed to train Multi-Binary environment agents.


### `make train-models`

Once the agents is trained , their models will be saved in zip files. Below commands are used to generate all the results.

### `make run-script `

For different loads, we evaluate the results and plot them by following command,

### `make loadvariation`

For different job lengths, we check behavior to see if all jobs are the same by following command,

### `make run-withholdjobs`

For different job lengths, we compare the slowdown by following command,

### `make run-raytune`

This command will run the hyperparameter optimization for both environments.


Happy Coding :)
