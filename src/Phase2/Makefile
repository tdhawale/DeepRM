#
# <aicon> build script.
#

.PHONY: all

#.PHONY: default test install uninstall clean realclean

# Make file for Scheduling Project
	
SHELL=/bin/bash
CONDAROOT = ~/miniconda3/bin/conda
prefix_discrete = Discrete_DeepRM/
prefix_multibinary = MultiBinary_DeepRM/

application_update:
	sudo apt-get update
	sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
	sudo apt-get install python-matplotlib

install_miniconda:
	@echo 'installing miniconda 3 '
	mkdir -p ~/miniconda3
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
	bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
	rm -rf ~/miniconda3/miniconda.sh
	~/miniconda3/bin/conda init bash
	~/miniconda3/bin/conda init zsh
	@echo 'installtion done Please restart your shell'

upgrade_miniconda:
	@echo 'updating the miniconda with the required prerequisites'
	~/miniconda3/bin/conda install python=3.7.6
	~/miniconda3/bin/conda install tensorflow=1.15 
	~/miniconda3/bin/conda install -c anaconda mpi4py 
	~/miniconda3/bin/conda install pytorch torchvision -c pytorch	
	@echo 'Done'	
	
pip_files:
	pip install gym
	pip install numpy --upgrade	
	pip install stable-baselines
	pip install SQLAlchemy
	cd $(prefix_discrete) ; pip install -e . 
	cd $(prefix_multibinary) ; pip install -e.

raytune:
	pip install ax-platform 
	pip install ray
	pip install ray[rllib]
	pip install cmake
	pip install cython 
	pip install git+https://github.com/cornellius-gp/gpytorch.git 
	pip install git+https://github.com/pytorch/botorch.git

lasagne_theano:
	pip install lasagne==0.1
	pip install -I theano==0.7

clean:
	@echo 'cleaning up temporary files'
	find . -name '*.pyc' -type f -exec rm {} ';'
	find . -name '__pycache__' -type d -print | xargs rm -rf
	@echo 'NOTE: you should clean up the following occasionally (by hand)'
	git clean -fdn


#Phase - I

run-script:
	python3 run.py
	python3 Script.py

run-stableBaseLine_Models_Training:
	python3 stableBaseLine_Models_Training.py
	
run-raytune:
	python3 raytuneAllAgents.py
   
loadvariation:
	python3 LoadVariation.py
	python3 loadVariationDiscrete.py

train-models:
	python3 Train_all.py

run-withholdjobs:
	python3 Withhold_Jobs.py

run-job-tradeoff:
	python3 Joblength_Tradeoff.py

#Phase - II
# Discrete Environment 

run-train:
	cd $(prefix_discrete) ; python3 train.py 
	cd $(prefix_discrete) ; python3 train.py --objective=Job_Completion_Time

run-raytune_discrete:
	cd $(prefix_discrete) ; python3 raytune_discrete.py

run-load_Variation_json:
	cd $(prefix_discrete) ; python3 load_Variation_json.py

run-load_Variation_Discrete:
	cd $(prefix_discrete) ; python3 load_Variation_Discrete.py

run-run:
	cd $(prefix_discrete) ; python3 run.py

run-Withheld_Jobs_Discrete:
	cd $(prefix_discrete) ; python3 Withheld_Jobs_Discrete.py

run-job_generation:
	cd $(prefix_discrete) ; python3 job_generation.py


#Multibinary 

run-Train_all:
	cd $(prefix_multibinary) ; python3 Train_all.py

run-LoadVariation:
	cd $(prefix_multibinary) ; python3 LoadVariation.py

run-Script:
	cd $(prefix_multibinary) ; python3 Script.py

run-Withhold_Jobs:
	cd $(prefix_multibinary) ; python3 Withhold_Jobs.py
