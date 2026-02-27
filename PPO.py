import numpy
from ppoagent import PPOAgent
from setup_logger import setup_logger
import shutil
import torch
import os
import numpy as np
from calc_performance2 import calc_performance
from calc_performance2_parallel import calc_performance_parallel
from ppo_actor import Actor_beta_dirichlet
from ppo_critic import Critic
import random

class PPO():
    def __init__(self, env, paramdf, meta):
        """
        PPO agent.
        env: Environment object
        paramdf: DataFrame containing hyperparameters for the agent
        meta: metadata (paramid, iteration, seed) for logging and saving results
        """
        self.paramdf = paramdf
        # define parameters
        ## env setup
        self.env = env
        ## meta info
        self.paramid = meta['paramid']
        self.iteration = meta['iteration']
        self.seed = meta['seed']
        ## define the path for the new directory
        self.parent_directory = "./PPO_results/"
        self.new_directory = f'seed{self.seed}_paramid{self.paramid}'
        self.path = os.path.join(self.parent_directory, self.new_directory)
        ## set path 
        os.makedirs(self.path, exist_ok= True)
        self.testwd = f'./PPO_results/{self.new_directory}'
        self.logger = setup_logger(self.testwd) ## set up logging

        # Device selection
        # options from paramdf['device']: {'cuda','gpu','cpu','mps'}
        requested_device = (
            str(self.paramdf['device']).lower()
            if isinstance(self.paramdf, dict) and 'device' in self.paramdf
            else 'auto'
        )

        if requested_device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        elif requested_device in ('cuda', 'gpu'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif requested_device == 'mps':
            self.device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print(f"Using {self.device.type} device")

        # parameters
        ## NN parameters
        self.state_size = len(self.env.obsspace_dim)
        self.action_size = len(self.env.actionspace_dim)
        self.actor_hidden_num = int(paramdf['actor_hidden_num']) # number of hidden layers in the actor network
        self.actor_hidden_size = eval(paramdf['actor_hidden_size']) # size of hidden layers in the actor network
        self.critic_hidden_num = int(paramdf['critic_hidden_num']) # number of hidden layers in the critic network for trunk
        self.critic_hidden_size = eval(paramdf['critic_hidden_size']) # size of hidden layers in the critic network for trunk

        ## training parameters
        self.rolloutlen = paramdf['rolloutlen'] # e.g., 20
        self.minibatch_size = paramdf['minibatch_size'] # e.g., 5 
        self.n_epochs = paramdf['n_epochs'] # e.g., 4
        self.max_steps = paramdf['max_steps'] # maximum steps per episode, e.g., 1000

        ## learning rates
        self.actor_lr = float(paramdf['actor_lr']) # learning rate for actor network
        self.critic_lr = float(paramdf['critic_lr']) # learning rate for critic network
        self.actor_lrdecayrate = float(paramdf['actor_lrdecay']) # learning rate decay rate for actor network
        self.critic_lrdecayrate = float(paramdf['critic_lrdecay']) # learning rate decay rate for critic network
        if paramdf['actor_minlr'] == 'inf':
            self.actor_min_lr = float('-inf') # minimum learning rate for actor network
        else:
            self.actor_min_lr = float(paramdf['actor_minlr'])
        if paramdf['critic_minlr'] == 'inf':
            self.critic_min_lr = float('-inf') # minimum learning rate for critic network
        else:
            self.critic_min_lr = float(paramdf['critic_minlr'])
        self.actor_lrdecaytype = paramdf['actor_lrdecaytype'] # learning rate decay type for actor network
        self.critic_lrdecaytype = paramdf['critic_lrdecaytype'] # learning rate decay type for critic network
        self.scheduler_info = eval(paramdf['scheduler_info'])

        ## standardize
        self.standardize = bool(int(paramdf['standardize'])) # whether to standardize the advantages

        ## loss coefficients 
        self.c1 = float(paramdf['c1']) # coefficient for value function loss
        self.c2 = float(paramdf['c2']) # coefficient for entropy bonus
        self.entropy_loss_included = bool(int(paramdf['entropy_loss_included'])) # whether to include entropy loss in the total loss
        self.policy_clip = float(paramdf['policy_clip']) # clipping parameter for PPO

        ## discounting and GAE lambda
        self.gamma = float(paramdf['gamma']) # discount factor
        self.gae_lambda = float(paramdf['gae_lambda']) # GAE lambda parameter

        # create networks
        if 'Hatchery3.3.' in self.env.envID:
            actor = Actor_beta_dirichlet(self.state_size, self.action_size+1, # add 1 for the beta parameter, which is the first output, and the rest are for the dirichlet distribution (total of 5 outputs for 4 actions)
                                        self.actor_hidden_size, self.actor_hidden_num,
                                        self.actor_lrdecayrate, self.actor_lr,
                                        self.actor_min_lr, self.actor_lrdecaytype, 
                                        self.scheduler_info, self.device)

        critic = Critic(self.state_size, 
                        self.critic_hidden_size, self.critic_hidden_num,
                        self.critic_lrdecayrate, self.critic_lr, 
                        self.critic_min_lr, self.critic_lrdecaytype, 
                        self.scheduler_info, self.device)
        
        # create agent
        self.agent = PPOAgent(c1=self.c1, c2=self.c2, entropy_loss=self.entropy_loss_included,  # loss coefficients
                        minibatch_size=self.minibatch_size,  # minibactch size
                        policy_clip=self.policy_clip, # PPO clipping parameter
                        gamma=self.gamma, gae_lambda=self.gae_lambda, # discount factor and GAE lambda
                        n_epochs=self.n_epochs, # number of epochs for updating the policy
                        actor=actor, critic=critic) # actor and critic networks
        self.episodenum = paramdf['episodenum']


    def train(self):
        best_score = 0
        score_history = []

        learn_iters = 0
        avg_score = 0
        n_steps = 0

        for i in range(self.episodenum):
            observation = self.env.reset()
            done = False
            score = 0
            episteps = 0
            while not done:
                action, prob, val = self.agent.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                n_steps += 1
                score += reward
                self.agent.remember(observation, action, prob, val, reward, done)
                if n_steps % self.rolloutlen == 0:
                    self.agent.learn()
                    learn_iters += 1
                observation = observation_
                episteps += 1
                if episteps >= self.max_steps:
                    break
            score_history.append(score)
            avg_score = numpy.mean(score_history[-100:])

            # is this really the best way to save?
            if avg_score > best_score:
                best_score = avg_score
                self.agent.save_models()

            print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score,
                'time_steps', learn_iters)
        x = [i+1 for i in range(len(score_history))] 