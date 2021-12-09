# AutoML algorithm search for different sparsity and latency

# Dependencies
- pyyaml == 5.3.0
- torch == 1.8.0
- torchvision == 0.9.0
- tensorflow >=2.4.0

# Pattern sets

- You can modify any pruning patterns in the rl_input.py and admm.py, admm40.py, admm60.py.

# How to run the experiment
	
	cd AutoML
	python rl_controller.py

-If you want to run the autoML on har dataset

	cd AutoML/har_all
	python rl_controller.py
