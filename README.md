# Model-Based Novelty Search (MBNS)

Warning: A cleanup of the repository needs to be performed, some environments dependancies are built upon private repositories that cannot be made public (will be changed)

This repository is built upon the Dynamics-Aware Quality-Diversity (DA-QD) [paper](https://arxiv.org/abs/2109.08522) original [repository](https://github.com/adaptive-intelligent-robotics/Dynamics-Aware_Quality-Diversity).
This repository adds compatibility with [OpenAI gym](https://github.com/openai/gym) environments and integrates several of them (can be checked in the [exps_utils.py](https://github.com/elias-hanna/mbns/blob/master/exps_utils.py) file).
This repository also adds an implementation of the Novelty Search algorithm, the Model-Based Novelty Search algorithm and the Zero-shot Diverse Archive Bootstrapping algorithm. It also contains drafts of other algorithms with local dynamics model approaches. 

## External Package Dependancies

- [fastsim_gym](https://github.com/elias-hanna/fastsim_gym)
- [mb_ge](https://github.com/elias-hanna/mb_ge)
- [redundant_arm](https://github.com/elias-hanna/redundant_arm) 
- diversity_algorithms_dev : to remove (private repository)

## Run the code

1. Create a directory $logs to store the results of the run
2. Run $algo on $env (on of the ones described in [exps_utils.py](https://github.com/elias-hanna/mbns/blob/master/exps_utils.py)) for $max_evals (real system evaluations):
  ```
  python3 gym_${algo}_main.py --log_dir $logs -e $env --max_evals $max_evals --num_cores 8 --dump_period 5000
  ```
3. Analyze results using the ```run_psm_analysis.sh``` script for policy search methods results (NS, MBNS, QD, DAQD), using the ```run_dab_analysis.sh``` script for 0-DAB results or using the ```run_transfer_analysis.sh``` script for model-based policy search methods transfer analysis results (MBNS, DAQD). Run from the results directory and change the variables inside the bash scripts to fit your architecture (results are printed over various repetitions by default).
  ```
  ./run_psm_analysis.sh
  ```
4. Other arguments:
- ```--model-horizon``` : int, change the prediction horizon on the learned dynamics model. Default is -1, which uses the environment task horizon.
- ```--n-waypoints``` : int, change the number of waypoints used in the behavioral descriptor of an individual, the last trajectory element projection in behavioral space is always used. Default is 1 (last trajectory element projection).
- ```--model-type``` : [det, det_ens, srf_ens, prob], det uses a deterministic neural network as learned dynamics model. det_ens uses an ensemble of deterministic neural networks as learned dynamics model. srf_ens uses an ensemble of spatial random fields as dynamics model (only for 0-DAB experiments). prob uses an ensemble of probabilistic neural networks (output is a probability distribution, parameterized by mean and variance, for each predicted dimension) as learned dynamics model.
- ```--ens-size``` : int, size of the dynamics model ensemble if one is used (det_ens, srf_ens and prob options for argument --model-type)
- ```--norm-controller-input``` : [0, 1], used only when the controller is a neural network, 1 normalizes the input dimensions using min-max normalization given the task. 
- ```--open-loop-control``` : [0, 1], used only when the controller is a neural network, 1 uses timesteps as an input to the controller.
- ```--pop-size``` : int, population size for the Novelty Search and Model-Based Novelty Search algorithm.
- ```--c-n-neurons``` : int, used only when the controller is a neural network, determines the number of neurons per hidden layer in controller (input layer dimension is always that of the state space, output laywer dimension is always that of the action space).
- ```--c-n-layers``` : int, used only when the controller is a neural network, determines the number of hidden layers in the controller.
- ```--c-type``` : [ffnn, rnn], used only when the controller is a neural network, ffnn uses a simple fully-connected feed-forward neural network as a controller, rnn uses a recurrent neural network.
- ```--arch-sel``` : [random, nov], selects individuals to add to the Novelty Search or Model-Based Novelty Search archive either randomly (random) or depending on their novelty (nov). 
- ```--model-ns-return``` : [archive, population, average_nov], selects the individuals to transfer onto the real system at the end of the model loop in Model-Based Novelty Search. archive returns the individuals added to the model archive, population returns the final population and average_nov returns the estimated most novel individuals on the model obtained throughout the whole model loop. 
- ```--adaptive-novl``` : change dynamically the novl parameter for the [Iso+Line](https://arxiv.org/pdf/1804.03906) mutation operator
- ```--transfer-err-analysis``` : run complementary analysis for behavioral descriptor estimation error
