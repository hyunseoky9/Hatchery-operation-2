import shutil
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
from ddpg_actor import Actor
from setup_logger import setup_logger
from stacking2 import *
from RunningMeanStd import RunningMeanStd
from calc_performance2 import calc_performance
from calc_performance2_parallel import calc_performance_parallel
class TD3():
    """Reinforcement Learning agent that learns using TD3."""
    def __init__(self, env, paramdf, meta):
        """
        env: Environment object that provides the state and action space.
        paramdf: DataFrame containing hyperparameters for the agent (Dict).
        meta: Metadata (paramid, iteration, seed) for the agent.
        """
        self.paramdf = paramdf
        # define parameters
        self.env = env
        # some path and logging settings    
        ## roll otu meta info
        self.paramid = meta['paramid']
        self.iteration = meta['iteration']
        self.seed = meta['seed'] 
        ## Define the path for the new directory
        self.parent_directory = './TD3 results'
        self.new_directory = f'seed{self.seed}_paramset{self.paramid}'
        self.path = os.path.join(self.parent_directory, self.new_directory)
        ## set path
        os.makedirs(self.path, exist_ok=True)
        self.testwd = f'./TD3 results/{self.new_directory}'
        self.logger = setup_logger(self.testwd) ## set up logging
        print(f'paramID: {self.paramid}, iteration: {self.iteration}, seed: {self.seed}')

        # device for pytorch neural network
        self.device = "cpu"
        print(f"Using {self.device} device")

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

        ## TD3 parameters
        self.policy_delay = int(paramdf['policy_delay'])  # delay term "d" for policy updates
        self.target_noise = float(paramdf['target_noise'])  # noise for target policy smoothing (standard deviation of Gaussian noise)
        self.target_noise_clip = float(paramdf['target_noise_clip']) # noise clipping for target policy smoothing

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
        self.critic_weight_decay = float(paramdf['critic_weight_decay'])

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

        ## Evaluation parameters
        self.evaluation_interval = int(paramdf['evaluation_interval'])
        self.performance_sampleN = int(paramdf['performance_sampleN'])
        self.parallel_testing = bool(int(paramdf['parallel_testing']))  # whether to use parallel testing or not
        ## Training length parameters
        self.max_steps = int(paramdf['max_steps'])
        self.episodenum = int(paramdf['episodenum'])

        # print out the parameters
        print(f'paramID: {self.paramid}, iteration: {self.iteration}, seed: {self.seed}')
        print(f"device: {self.device}")
        print(f"state size: {self.state_size}, action size: {self.action_size}")
        print(f"actor: hidden num: {self.actor_hidden_num}, hidden size: {self.actor_hidden_size}")
        print(f"critic: action hidden num: {self.critic_action_hidden_num}, action hidden size: {self.critic_action_hidden_size}")
        print(f"critic: state hidden num: {self.critic_state_hidden_num}, state hidden size: {self.critic_state_hidden_size}")
        print(f"critic: trunk hidden num: {self.critic_trunk_hidden_num}, trunk hidden size: {self.critic_trunk_hidden_size}")
        print(f"actor lr: {self.actor_lr}, actor lr decay rate: {self.actor_lrdecayrate}, actor min lr: {self.actor_min_lr}")
        print(f"critic lr: {self.critic_lr}, critic lr decay rate: {self.critic_lrdecayrate}, critic min lr: {self.critic_min_lr}")
        print(f"critic weight decay: {self.critic_weight_decay}")
        print(f"standardize: {self.standardize}")
        print(f"exploration: mu: {self.exploration_mu}, theta: {self.exploration_theta}, sigma: {self.exploration_sigma}")
        print(f"buffer size: {self.buffer_size}, batch size: {self.batch_size}")
        print(f"policy delay: {self.policy_delay}, target noise: {self.target_noise}, target noise clip: {self.target_noise_clip}")
        print(f"gamma: {self.gamma}, tau: {self.tau}")
        print(f"fstack: {self.fstack}")
        print(f"evaluation interval: {self.evaluation_interval}, performance sampleN: {self.performance_sampleN}")
        print(f"max steps: {self.max_steps}, episodenum: {self.episodenum}")

        ##############################################
        # setting up the network, exploration noise, and replay buffer

        ## Actor (Policy) Model
        self.actor_local = Actor(self.state_size*self.fstack, self.action_size, self.actor_hidden_size, self.actor_hidden_num,
                                  self.actor_lrdecayrate, self.actor_lr).to(self.device)
        self.actor_target  = copy.deepcopy(self.actor_local).to(self.device)

        ## Critic (Value) Model
        self.critic1_local = Critic(self.state_size*self.fstack, self.action_size, self.critic_state_hidden_size, self.critic_state_hidden_num,
                                    self.critic_action_hidden_size, self.critic_action_hidden_num, self.critic_trunk_hidden_size,
                                    self.critic_trunk_hidden_num, self.critic_lrdecayrate, self.critic_lr).to(self.device)
        self.critic2_local = Critic(self.state_size*self.fstack, self.action_size, self.critic_state_hidden_size, self.critic_state_hidden_num,
                                    self.critic_action_hidden_size, self.critic_action_hidden_num, self.critic_trunk_hidden_size,
                                    self.critic_trunk_hidden_num, self.critic_lrdecayrate, self.critic_lr).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1_local).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2_local).to(self.device)

        ## Optimizers
        self.critic1_opt = torch.optim.Adam(self.critic1_local.parameters(), lr=self.critic_lr, weight_decay=self.critic_weight_decay, eps=1e-8)
        self.critic2_opt = torch.optim.Adam(self.critic2_local.parameters(), lr=self.critic_lr, weight_decay=self.critic_weight_decay, eps=1e-8)
        self.actor_opt = torch.optim.Adam(self.actor_local.parameters(), lr=self.actor_lr, eps=1e-8)

        ## Schedulers
        self.critic1_scheduler = ExponentialLR(self.critic1_opt, gamma=self.critic_lrdecayrate)  # Exponential decay
        self.critic2_scheduler = ExponentialLR(self.critic2_opt, gamma=self.critic_lrdecayrate)  # Exponential decay
        self.actor_scheduler = ExponentialLR(self.actor_opt, gamma=self.actor_lrdecayrate)  # Exponential decay

        ## initialize step counter for delayed updates
        self.learn_step = 0


        # Freeze target nets so their params aren’t updated by optim.step()
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        ## standardization
        self.rms = RunningMeanStd(len(env.obsspace_dim), self.max_steps) if self.standardize else None

        ## Noise process
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        ## Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

    def reset_episode(self):
        self.noise.reset()
        state, _ = self.env.reset()
        self.last_state = state*self.fstack
        return state

    def storeNlearn(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
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
        with torch.no_grad():
            logits_next = self.actor_target.body(next_states) # shape [B, K]
            noise = torch.normal(mean=0.0,std=self.target_noise,size=logits_next.shape,device=logits_next.device) # Gaussian exploration noise
            noise = noise.clamp(-self.target_noise_clip,self.target_noise_clip) # clip the noise element-wise
            actions_next = torch.softmax(logits_next + noise, dim=-1)  # shape [B, K]
            q1_targets_next = self.critic1_target(next_states, actions_next) # there should be 2 critics.
            q2_targets_next = self.critic2_target(next_states, actions_next) # there should be 2 critics.
            q_targets      = rewards + self.gamma * torch.min(q1_targets_next, q2_targets_next) * (1 - dones)

        # critic update
        q1_expected   = self.critic1_local(states, actions)
        q2_expected   = self.critic2_local(states, actions)
        critic1_loss  = F.mse_loss(q1_expected, q_targets)
        critic2_loss  = F.mse_loss(q2_expected, q_targets)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        self.critic1_scheduler.step() # Decay the learning rate
        if self.critic_min_lr != float('-inf'): # if there's a minimum learning rate, don't go below it.
            if self.critic1_opt.param_groups[0]['lr'] < self.critic_min_lr:
                self.critic1_opt.param_groups[0]['lr'] = self.critic_min_lr
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
        self.critic2_scheduler.step() # Decay the learning rate
        if self.critic_min_lr != float('-inf'): # if there's a minimum learning
            if self.critic2_opt.param_groups[0]['lr'] < self.critic_min_lr:
                self.critic2_opt.param_groups[0]['lr'] = self.critic_min_lr
        self.learn_step += 1

        # delayed update for actor and target networks
        if self.learn_step % self.policy_delay == 0:
            # actor update
            actions_pred = self.actor_local(states)
            actor_loss   = -self.critic1_local(states, actions_pred).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self.actor_scheduler.step() # Decay the learning rate
            if self.actor_min_lr != float('-inf'):
                if self.actor_opt.param_groups[0]['lr'] < self.actor_min_lr:
                    self.actor_opt.param_groups[0]['lr'] = self.actor_min_lr

            # 5. Soft-update target nets 
            self.soft_update(self.critic1_local, self.critic1_target, self.tau)
            self.soft_update(self.critic2_local, self.critic2_target, self.tau)
            self.soft_update(self.actor_local,  self.actor_target,  self.tau)

    def soft_update(self, local_net, target_net, tau):
        """theta_target ← tau theta_local  +  (1-tau) theta_target"""
        for l_param, t_param in zip(local_net.parameters(), target_net.parameters()):
            t_param.data.mul_(1.0 - tau).add_(l_param.data, alpha=tau)

    def train(self):
        """Train the agent, test the trained agent every x episodes,
        and save the checkpoint models and the final & best model"""
        n_episodes = self.episodenum
        max_t = self.max_steps  # max steps per episode
        fstack = self.fstack  # frame stacking
        scores = [] # online scores
        inttestscores = [] # interval test scores
        for i_episode in range(1, n_episodes + 1):
            state = self.reset_episode()  # this resets both env and internal noise
            score = 0.0
            state = self.rms.normalize(self.env.obs) if self.standardize else self.env.obs
            state = np.concatenate([state] * fstack)
            for t in range(max_t):
                action = self.actor_local.act(state,self.noise)  # get action from actor (with noise for exploration)
                next_state, reward, done, _ = self.env.step(action)  # step in the env
                next_state = self.rms.normalize(next_state) if self.standardize else next_state # standardize
                if self.standardize: # standardize
                    self.rms.stored_batch.append(next_state) # store the state for running mean std calculation
                    next_state = self.rms.normalize(next_state)
                    if self.rms.rolloutnum >= self.rms.updateN:
                        self.rms.update()
                    self.rms.rolloutnum += 1
                next_state = stacking(self.env, state, next_state) # stack
                self.storeNlearn(action, reward, next_state, done)  # adds transition to memory and learns from sampling
                state = next_state
                score += reward
                if done:
                    break
            scores.append(score)
            if i_episode % 100 == 0:
                print(f"Episode {i_episode}\tScore: {score:.2f}")

            if i_episode % self.evaluation_interval == 0: # calculate average reward every evaluation interval episodes
                if self.parallel_testing:
                    inttestscore = calc_performance_parallel(self.env, self.device, self.seed, self.paramdf['envconfig'], self.rms, self.fstack, self.actor_local, self.performance_sampleN, self.max_steps)
                else:
                    inttestscore = calc_performance(self.env,self.device,self.rms,self.fstack,self.actor_local,self.performance_sampleN,self.max_steps)
                inttestscores.append(inttestscore)
                torch.save(self.actor_local, f"{self.testwd}/PolicyNetwork_{self.env.envID}_par{self.env.parset}_dis{self.env.discset}_TD3_episode{i_episode}.pt")
                # save the running mean and sd/var as well for this episode in pickle
                if self.standardize:
                    with open(f"{self.testwd}/rms_{self.env.envID}_par{self.env.parset}_dis{self.env.discset}_TD3_episode{i_episode}.pkl", "wb") as file:
                        pickle.dump(self.rms, file)
                critic1_current_lr = self.critic1_opt.param_groups[0]['lr']
                critic2_current_lr = self.critic2_opt.param_groups[0]['lr']
                actor_current_lr = self.actor_opt.param_groups[0]['lr']
                print(f"Episode {i_episode}, Learning Rate: A{actor_current_lr}/C1{critic1_current_lr}/C2{critic2_current_lr} Avg Performance: {inttestscore:.2f}")
                print('-----------------------------------')

        print('calculating the average reward with the final Q network')
        if self.parallel_testing:
            final_testscore = calc_performance_parallel(self.env, self.device, self.seed, self.paramdf['envconfig'], self.rms, self.fstack, self.actor_local, self.performance_sampleN, self.max_steps)
        else:
            final_testscore = calc_performance(self.env,self.device,self.rms,self.fstack,self.actor_local,self.performance_sampleN,self.max_steps)
        inttestscores.append(final_testscore)
        print(f'final average reward: {final_testscore:.2f}')
        torch.save(self.actor_local, f"{self.testwd}/PolicyNetwork_{self.env.envID}_par{self.env.parset}_dis{self.env.discset}_TD3_episode{i_episode}.pt")
        if self.standardize:
            with open(f"{self.testwd}/rms_{self.env.envID}_par{self.env.parset}_dis{self.env.discset}_TD3_episode{i_episode}.pkl", "wb") as file:
                pickle.dump(self.rms, file)

        # save results and performance metrics.
        ## save last model
        torch.save(self.actor_local, f"{self.testwd}/PolicyNetwork_{self.env.envID}_par{self.env.parset}_dis{self.env.discset}_TD3.pt")
        if self.standardize:
            with open(f"{self.testwd}/rms_{self.env.envID}_par{self.env.parset}_dis{self.env.discset}_TD3.pkl", "wb") as file:
                pickle.dump(self.rms, file)

        ## save best model
        bestidx = np.array(inttestscores).argmax()
        bestfilename = f"{self.testwd}/PolicyNetwork_{self.env.envID}_par{self.env.parset}_dis{self.env.discset}_TD3_episode{(bestidx+1)*self.evaluation_interval}.pt"
        print(f'best Policy network found at episode {(bestidx+1)*self.evaluation_interval}')
        shutil.copy(bestfilename, f"{self.testwd}/bestPolicyNetwork_{self.env.envID}_par{self.env.parset}_dis{self.env.discset}_TD3.pt")

        ## save performance
        np.save(f"{self.testwd}/rewards_{self.env.envID}_par{self.env.parset}_dis{self.env.discset}_TD3.npy", inttestscores)

        ## lastly save the configuration.
        param_file_path = os.path.join(self.testwd, f"config.txt")
        with open(param_file_path, 'w') as param_file:
            for key, value in self.paramdf.items():
                param_file.write(f"{key}: {value}\n")

        return self.actor_local, inttestscores