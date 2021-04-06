This repository will include materials I used when writing my master thesis.

The code has been tested to work on Ubuntu with CUDA enabled and on Arch with CPU-only mode.

-----------------------------------------------------------------

Quick startup:

-----------------------------------------------------------------
1. INSTALL
-----------------------------------------------------------------
Python requirements (except pytorch) can be installed with:

	pip3 install -r requirements.txt

For full install of all neccessary components run:

	bash install.sh

Notes:

	Before running install.sh install (or make sure it is installed) swig and cmake.

	If git is installed on your system and you run the code from a git repository,
	the repository's hash will be logged

	If you don't install lambda stack during install then make sure you have pytorch (CPU or GPU are both supported)

	Use of anaconda or similar environment management software is advised

-----------------------------------------------------------------
2. RUN TRAINING
-----------------------------------------------------------------
Navigate to:

	code/nextro_training/minitaur_inspired_agent

Run single training instance:

	./agent_minitaur_inspired.py --mode=train --frames=500000 --man_mod

*This will run 500k frames of training with manually chosen parameters

Run training batch:

	./train_batch.sh 7 100000

*This will sequentially run 7 training sessions of 100k frames each, with randomized reward function coeficients.

-----------------------------------------------------------------
3. RUN TESTING
-----------------------------------------------------------------
Navigate to:

	code/nextro_training/minitaur_inspired_agent

Run:

	./agent_minitaur_inspired.py --mode=test --episodes=10 --loc=best/long_train --render

*This will load a pretrained example network, render the environment and run 10 test episodes


-----------------------------------------------------------------
-----------------------------------------------------------------
For other details on how to use the code and hwo it works see text/thesis.pdf
-----------------------------------------------------------------
-----------------------------------------------------------------
