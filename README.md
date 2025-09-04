# Hatchery operation project
## Code Scripts

### environment classes

Toy RGSM hatchery augmentation environments:
mock models
- *env0_0.py*: mock model of env1.0 made for practicing tabular Q learning. Practically useless now.
- *env1_0.py*: first mock model for hatchery operation and population dynamics
- *env1_1.py*: continuous state space from env1.0
- *env1_2.py*: continuous state & action space from env1.0
- *env2_0.py*: partial observability added from env1.0
- *env2_1.py*: POMDP format but fully observable. (basically env1.0)
- *env2_2.py*: same as 2.1 but past population size is not observable.
- *env2_3.py*: same as env2.1 but observation but NWm1 is not observed AND NW is observed only in Fall.
- *env2_4.py*: same as Env2.0 but catch (y) is observed every season (both in fall and spring) (most accurate of 2.x)
- *env2_5.py*: same as Env2.0 but catch (y) is observed every season and is continuous (both in fall and spring)
real-deals
- *Hatchery3_0.py*: environment for the hatchery problem implementation-grade parameterization. uses Allelic richness as a genetic diversity variable and it affects the reproduction process. (deprecated)
- *Hatchery3_1.py*:  environment for the hatchery problem implementation-grade parameterization. uses heterozygosity as a genetic diversity variable but it does not affect the demographic process unlike Hatchery3_0. (useful for assessing heterozygosity dynamics)
- *Hatchery3_2.py*: uses effective popualtion size. Still has to take spring production action and stocking actions have to be taken one reach at a time, and it's for Q-learning. (not that useful)
- *Hatchery3_2_2.py*: Same as hatchery 3.2, but all the fall actions are taken at once, and spring decision is not taken. MAIN MODEL USED FOR THE PROJECT. You can incorporate uncertainty in the parameters as well sampling from the posterior distribution.
- *Hatchery3_2_3.py*: Same as hatchery 3.2.2, but all the population parameters are observable.
- *Hatchery3_2_4.py*: Same as hatchery 3.2.2, but simulation does not terminate anymore. If the subpopulation goes below its local threshold, the subpopulation will fall to 0. Even all subpopulation goes to 0, the populatin will keep going
  
Tiger POMDP (Chades et al. 2008)
- *tiger.py*: Tiger POMDP used for testing DRQN
  
*there are matlab versions (.m) of some of these environments*


### Algorithm running / plotting and analyzing outputs
- *env1.0_performance_testing.ipynb*: compute average reward for a given policy
- *env1.0_plotting_running.ipynb*: running optimization algorithms (value iteration, Tabular Q learning, DeepQN, etc.) and then analyzing/plotting the results for env1.0
- *env0.0_plotting_running.ipynb*: same as above but for env0.0
(environment name)_performance_testing.ipynb is for evaluating the the named environment, and (environment name)_plotting_running.ipynb is training the named env.

### Additional notebooks
- *effective population size calculation.ipynb*: effective population size calculation note
- *effective population size validation.ipynb*: validation of the effective population size model to the actual LDNe data from Osborne et al. 2024 using the population estimates from Yackulic et al. 2023
### Optimization algorithms
#### value iteration
- *value_iteration.py*: perform value iteration
  
#### tabular Q learning
- *td_learning_nonclass.py*: perform Q learning or sarsa on the model

#### Deep Q Network (Rainbow)
- *Rainbow.py*: main DQN script. Has all the features in Rainbow.
- *choose_action.py*: choose action from Q or randomly depending on epsilon 
- *distributionalRL.py*: distributional RL feature in rainbow
- *DuelQNN.py*: Dueling network
- *NoisyLinear.py*: Noisy network
- *nq.py*: Queue for N-step learning
- *PrioritizedMemory.py*: Prioritized memory buffer
- *QNN.py*: Vanilla Q network
- *pretrain.py*: fills up the replay buffer before starting the environment simulation.

#### DRQN
- *DRQN.py*: main DRQN script. shares some of the function friles with DQN
- *RQNN.py*: network used for DRQN. has LSTM in it. 
- *RQNN.py*: similar to RQNN but LSTM layers come before the FF layers. Don't use it much.

#### TD3 & DDPG:
- *TD3.py*: main TD3 script
- *DDPG.py*: main DDPG script. not used much because TD3 is a better version of TD3.
- *ddpg_critic.py*: critic network class for Critic. mostly deprecated in this project because critic2 worked better.
- *ddpg_critic2.py*: critic network class for Critic2 Unlike Critic, this version merges state and action features from the start. used for td3 too
- *ddpg_actor.py*: actor network class. used for td3 too.
- *OUNoise.py*: Ornstein–Uhlenbeck process noise for exploration of td3 and ddpg.
- *ReplayBuffer.py*: replay buffer class for ddpg and td3.

#### related scripts for different RL tactics
- *stacking.py*: framestacking. used in Q-learning
- *stacking2.py*: framestacking code used in td3 and ddpg.
- *RunningMeanStd.py*: standardizing class object. updates the mean and std for standardization during the training simulation dynamically.
- *FixedMeanStd.py*: standardizing class object. Unlike runningmeanstd, the mean and std are fixed. Fixed values were obtained from independent simulations not taking any actions. works better than running mean std, so this is used for all td3 training for this proj. s
- *absorbing.py*: function that returns true if the input state is an absorbing, simulation-terminating state.
#### Bash scripts
*tiger_performance_printout_search.py*: goes through the .out files created from the HPC jobs and finds episodes that has high performance with certain policy attributes (e.g., policy has both managing and surveying). 
*env2_x_performance_printout_search.py*: same as above but for env2.x's.

### Hydrological model files
LC_GAM_(reach).pkl - *(reach=Isleta,ABQ, Otowi, San Acacia) - GAM model that predicts the Larval Carrying capacity index (LC) for an input springflow for respective reach.
AR1_normalized.py - AR1 model for simulating springflow annually. A newer version of AR1.py that goes through transformation processes (normalization+logit transformation) that keeps the prediction within historical bounds (min - 0.1*min , max + 0.1*max)
ar1_normalized_params.pkl - parameter object files needed to run AR1_normalized
AR1.py - AR1 model for simulating springflow annually

### Others
Performance evaluation scripts
*calc_performance.py*: Calculates expected reward by environment simulation, given Q or policy.
*calc_performance2.py*: Calculates expected reward by environment simulation, for Td3/ddpg.
*calc_performance2_parallel*: does the same as above but parallelized to make it efficient.
*performance_tester.py*: Uses calc_performance.py to calculate performance, used in tandem with RL algorithms (e.g., Rainbow.py). RL algorithms usually run this in a separate process for efficiency. (only for Q learning)
*performance.m*: calculates avg perofrmance with a given environment (currently env2.0)

*decay curves.R*: made visualizing different types of decay curves to decide on decay curve parameters. Used for epislon in epsilon-greedy and learning rate.

Scripts for running the training session in (env)_plotting_running.ipynb:
*call_paramset.py*: handles initializing the environment in the training session.

logging
*setup_logger.py*: Set up a logger and redirect print() output to it

Scripts made for practicing function approximation using neural net.
- *fn_approx_practice.ipynb*
- *fnapproxto.py*
- *function_approximation_datagen.R*
- *lstm practice/lotka volterra.ipynb*: for practicing lstm

## Input Files
Parameterization of the environments 
- parameterization_env0.0.csv
- parameterization_env1.0.csv
- parameterization_hatchery3.0.csv
- parameterization_hatchery3.1.csv - for 3.1 and 3.2.x.
Parmaeter posterior sample files
- uncertain_parameters_posterior_samples4POMDP.csv: posterior samples derived from the Yackulic IPM that I ran using int3re.stan using pseudocode4Hyun_lab.R (in /hatchery operation/codes/popmodel/genetic_var_added). 
- 2021-25 spawning summary.xlsx - information on the number of stock-ready-fish produced per females for BIOPARK. Used Ponded/female values in the first tab after multiplying it by 0.8 (as Thomas A. recommended) and averaging the values across years. The file itself is not directly used, but this value is used in defining the variable Nb in the Hatchery3.2.2 environment script. 
- Dexter RGSM spawning.xlsx - same thing as above but for Dexter. Only for 2025. Divide "Expected number of fish to harvest" by "females that spawned" to get the number of stock-ready-fish produced per female. (800)
- Osborne et al 2024 LDNe (Fig4b).csv - mean and CI values for LDNe measured from Osborne et al. 2024 (fig4b) using WebPlotDigitizer. 
- Osborne et al. 2024 _fib4b_.tar - contains file that you can load on WebPlotDigitizer


## Output Files
### Q, V, policy outputs
State action function, Value function, and Policy results made by RL algorithms
- policy, Q, V function outputs (tabular ones)
format: 
{policy,Q, or V}_{model version}\_par{parameterization id}\_dis{discretization id}\_{optimization algorithm used}.pkl

- Q function in network.
QNetwork_{model version}\_par{parameterization id}\_dis{discretization id}\_{optimization algorithm used}.pkl

- TD3 function in network
PolicyNetwork_Hatchery(version)_par(env parameter version)_dis(discretization version. -1 most of the time)_TD3_episode(checkpoint episode #).pt - checkpoint models
PolicyNetwork_Hatchery(version)_par(env parameter version)_dis(discretization version. -1 most of the time)_TD3_episode(checkpoint episode #).pt - best of all the checkpoints
rms_Hatchery(version)_par(env param version)_dis(discretization ver)_TD3_episode(checkpoint epi#).pkl - standardization obj that needs to be run together when using the network in a simulation for evaluation or for anything else. rms means running mean std. It's most likely from the fixed FixedMeanStd.py, which has fixed rms. RunningMeanStd.py never worked very well.

These files are results folders where trained networks are in. 
In each of these folders, there's /good_ones, where models that are selected by hand are in. 
  - *td_results*: results from tabular Q learning
  - *value iter results*: results from value iteration
  - *deepQN results*: results from deep Q learning
  - *DRQN results*: results from DRQN, has parameter id and seed number used for the result.
    - *HPCoutput*: saved HPC printouts.
    - *env2_1_good_ones*: saved network outputs on env2.1.
  - *TD3 results*: results from TD3. Each folder (seed#_paramset#) is an instance of training session.
  - *DDPG results*: results from DDPG. Almost never used because I just went to use TD3 instead.
goodone_mover.py - python script that you run on terminal to move the selected trained network folders to /good_ones in a respective results folder. You specify the seeds you selected as an input. You have to also specify the hyperparamset (paramset) in the script. (e.g., python goodone_mover.py [1212,24242,23232])

### Auxillary outputs
These are outputs that aren't Q,V, or policy outputs but outputs made by the algorithm scripts that helps the algorithm calculate the policy or value functions efficiently. 

- initQopt2_{envID}_par{parID}_dis{discID}.pkl: initial Q for tabular Q learning with initialization option 2. Check the td learning portion of env1.0_plotting_running.ipynb for details on initialization

- initQopt3_{envID}_par{parID}_dis{discID}.pkl:initial Q for tabular Q learning with option 3.
- trp_{envID}_par{parID}_dis{discID}.pkl: saved transition probability for the environment used for value iteration. Took one day of simulating transitions to make this.
- {state var}prob(tau_)_{envID}_par{parID}_dis{discID}.pkl: saved transition probability for each state variable (e.g. H, NW) in different tau (season). Made for value iteration for env1.0, but not used because calculating transition probability by state variables did not do very well in getting expected transition probabilities.

- persistence_probabilities_Hatchery3.2.2.pkl - persistence probability for every different parameter sample from the posterior distribution.

## Environments.
- *Env0.0*: Second environment model made. Drastically simplified version of Env1.0. It was made to practice the tabular Q learning. Now that I've moved on from tabular Q learning, it is no longer relevant.
- *Env1.0*: First environmental model ever made. Has genetic and demographic component of the augmentation environment. See document in overleaf.
- *Env1.1*: 


## Hyperparameter sets (in /hyperparamsets)

Collection of hyperparameters used to run Deep Q network and other network based algorithms (policy gradient). Shows performance of the set used as well.

- (environment) DQNbest.csv: hyperparameter settings used for DQN (rainbow)
- (environment) TD3bests.csv: hyperparameter settings used for TD3
- (environment) DDPGbests.csv: hyperparameter settings used for DDPG


## ETC.
when making new environmnet.txt: when you make a new environment, make sure those environments are registerd in the python files listed in this text file.