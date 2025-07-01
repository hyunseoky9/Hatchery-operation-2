from ddpg_actor import Actor
from ddpg_critic import Critic
from OUNoise import OUNoise
from ReplayBuffer import ReplayBuffer
import random
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
import pickle
import os
import copy
from setup_logger import setup_logger
from stacking import *
from RunningMeanStd import RunningMeanStd

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, env, paramdf, meta):
        """
        env: Environment object that provides the state and action space.
        paramdf: DataFrame containing hyperparameters for the agent (Dict).
        meta: Metadata (paramid, iteration, seed) for the agent.
        """
        # define parameters
        self.env = env
        # some path and logging settings    
        ## roll otu meta info
        self.paramid = meta['paramid']
        self.iteration = meta['iteration']
        self.seed = meta['seed'] 
        ## Define the path for the new directory
        self.parent_directory = './DDPG results'
        self.new_directory = f'seed{self.seed}_paramset{self.paramid}'
        self.path = os.path.join(self.parent_directory, self.new_directory)
        ## set path
        os.makedirs(self.path, exist_ok=True)
        self.testwd = f'./DDPG results/{self.new_directory}'
        self.logger = setup_logger(self.testwd) ## set up logging
        print(f'paramID: {self.paramid}, iteration: {self.iteration}, seed: {self.seed}')

        # device for pytorch neural network
        device = "cpu"
        print(f"Using {device} device")

        # parameters
        ## NN parameters
        self.state_size = len(env.obsspace_dim)
        self.action_size = len(env.actionspace_dim)

        self.actor_hidden_num = int(paramdf['actor_hidden_num'])
        self.actor_hidden_size = eval(paramdf['actor_hidden_size'])
        self.critic_action_hidden_num = int(paramdf['critic_action_hidden_num'])
        self.critic_action_hidden_size = eval(paramdf['critic_action_hidden_size'])
        self.critic_state_hidden_num = int(paramdf['critic_state_hidden_num'])
        self.critic_state_hidden_size = eval(paramdf['critic_state_hidden_size'])
        self.critic_trunk_hidden_num = int(paramdf['critic_trunk_hidden_num'])
        self.critic_trunk_hidden_size = eval(paramdf['critic_trunk_hidden_size'])

        ## Learning rate
        self.actor_lr = float(paramdf['actor_lr'])
        self.critic_lr = float(paramdf['critic_lr'])
        self.actor_lrdecayrate = float(paramdf['actor_lrdecay'])
        self.critic_lrdecayrate = float(paramdf['critic_lrdecay'])
        if paramdf['actor_minlr'] == 'inf':
            self.actor_min_lr = float('-inf')
        else:
            self.actor_min_lr = float(paramdf['actor_minlr'])
        if paramdf['critic_minlr'] == 'inf':
            self.critic_min_lr = float('-inf')
        else:
            self.critic_min_lr = float(paramdf['critic_minlr'])
        ## weight decay
        self.critic_weight_decay = float(paramdf['weight_decay'])

        ## standardize
        self.standardize = bool(int(paramdf['standardize']))

        ## Noise process
        self.exploration_mu = float(paramdf['exploration_mu'])
        self.exploration_theta = float(paramdf['exploration_theta'])
        self.exploration_sigma = float(paramdf['exploration_sigma'])

        ## Replay buffer memory parameters
        self.buffer_size = int(paramdf['buffer_size'])  # size of the replay buffer
        self.batch_size = int(paramdf['batch_size'])  # size of each training batch

        ## Algorithm parameters
        self.gamma = env.gamma  # discount factor
        self.tau = float(paramdf['tau'])  # for soft update of target parameters
        self.fstack = int(paramdf['framestacking'])# framestacking

        ## Training length parameters
        self.evaluation_interval = int(paramdf['evaluation_interval'])
        self.performance_sampleN = int(paramdf['performance_sampleN'])
        self.max_steps = int(paramdf['max_steps'])
        self.episodenum = int(paramdf['episodenum'])

        ##############################################
        # setting up the network, exploration noise, and replay buffer

        ## Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.actor_hidden_size, self.actor_hidden_num,
                                  self.actor_lrdecayrate, self.actor_lr)
        self.actor_target  = copy.deepcopy(self.actor_local)   # fast one-liner

        ## Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, self.critic_state_hidden_size, self.critic_state_hidden_num,
                                    self.critic_action_hidden_size, self.critic_action_hidden_num, self.critic_trunk_hidden_size,
                                    self.critic_trunk_hidden_num, self.critic_lrdecayrate, self.critic_lr)
        self.critic_target = copy.deepcopy(self.critic_local)   # fast one-liner

        ## Optimizers 
        self.critic_opt = torch.optim.Adam(self.critic_local.parameters(), lr=self.critic_lr,weight_decay=self.critic_weight_decay, eps=1e-8)
        self.actor_opt = torch.optim.Adam(self.actor_local.parameters(), lr=self.actor_lr, eps=1e-8)

        ## Schedulers
        self.critic_scheduler = ExponentialLR(self.critic_opt, gamma=self.critic_lrdecayrate)  # Exponential decay
        self.actor_scheduler = ExponentialLR(self.actor_opt, gamma=self.actor_lrdecayrate)  # Exponential decay


        # Freeze target nets so their params aren’t updated by optim.step()
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        ## standardization
        rms = RunningMeanStd(len(env.obsspace_dim), self.max_steps) if self.standardize else None

        ## Noise process
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        ## Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

    def reset_episode(self):
        self.noise.reset()
        state = self.env.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        device = self.device                        # e.g. torch.device("cuda")

        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states      = torch.as_tensor(np.vstack([e.state for e in experiences]),dtype=torch.float32, device=device)
        actions     = torch.as_tensor(np.vstack([e.action for e in experiences]),dtype=torch.float32, device=device)
        rewards     = torch.as_tensor(np.vstack([e.reward for e in experiences]),dtype=torch.float32, device=device)
        dones       = torch.as_tensor(np.vstack([e.done for e in experiences]),dtype=torch.float32, device=device)      # or dtype=torch.bool
        next_states = torch.as_tensor(np.vstack([e.next_state for e in experiences]),dtype=torch.float32, device=device)

        # critic target 
        with torch.no_grad():                                   # <– no grads here
            actions_next   = self.actor_target(next_states)     # μ′(s′)
            q_targets_next = self.critic_target(next_states, actions_next)
            q_targets      = rewards + self.gamma * q_targets_next * (1 - dones)
        
        # critic update
        q_expected   = self.critic_local(states, actions)
        critic_loss  = F.mse_loss(q_expected, q_targets)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # actor update
        actions_pred = self.actor_local(states)
        actor_loss   = -self.critic_local(states, actions_pred).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 5. Soft-update target nets
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local,  self.actor_target,  self.tau)

    def soft_update(self, local_net, target_net, tau):
        """theta_target ← tau theta_local  +  (1-tau) theta_target"""
        for l_param, t_param in zip(local_net.parameters(), target_net.parameters()):
            t_param.data.mul_(1.0 - tau).add_(l_param.data, alpha=tau)

    def train(self):
        """Train the agent"""
        n_episodes = self.episodenum
        max_t = self.max_steps  # max steps per episode
        scores = []

        for i_episode in range(1, n_episodes + 1):
            state = self.reset_episode()  # this resets both env and internal noise
            score = 0.0

            for t in range(max_t):
                action = self.actor_local.act(state)  # get action from actor (with noise for exploration)
                next_state, reward, done, _ = self.env.step(action)  # step in the env

                self.step(action, reward, next_state, done)  # this triggers learning
                state = next_state
                score += reward

                if done:
                    break

            scores.append(score)
            print(f"Episode {i_episode}\tScore: {score:.2f}")