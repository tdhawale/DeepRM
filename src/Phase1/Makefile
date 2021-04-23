#
# <aicon> build script.
#

.PHONY: all

#.PHONY: default test install uninstall clean realclean

# Need to specify bash in order for conda activate to work.
#SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
#CONDA_ACTIVATE=source ~/miniconda3/etc/profile.d/conda.sh ; conda activate ; conda activate

#py3build:
#	($(CONDA_ACTIVATE) aicon)


# Oneshell means I can run multiple lines in a recipe in the same shell, so I don't have to
# chain commands together with semicolon
	
SHELL=/bin/bash
CONDAROOT = ~/miniconda3/bin/conda

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
	@echo 'Done'	
	
pip_files:
	pip install gym
	pip install numpy --upgrade	
	pip install stable-baselines
	pip install SQLAlchemy
	pip install -e .
	
raytune:
	pip install ax-platform 
	pip install ray
	pip install ray[rllib]
	pip install cmake
	pip install cython

clean:
	@echo 'cleaning up temporary files'
	find . -name '*.pyc' -type f -exec rm {} ';'
	find . -name '__pycache__' -type d -print | xargs rm -rf
	@echo 'NOTE: you should clean up the following occasionally (by hand)'
	git clean -fdn

realclean: clean rebuild

run-script:
	python3 run.py
	python3 Script.py

run-stableBaseLine_Models_Training:
	python3 stableBaseLine_Models_Training.py
	
run-raytune:
	python3 raytuneAllAgents.py
	python3 raytuneAllAgentsMB.py
   
loadvariation:
	python3 LoadVariation.py
	python3 loadVariationDiscrete.py

train-models:
	python3 Train_all.py

run-withholdjobs:
	python3 Withhold_Jobs.py

run-job-tradeoff:
	python3 Joblength_Tradeoff.py
