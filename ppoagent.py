import os 
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim


class PPOMemory:
    def __init__(self, minibatch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.entropies = []
        self.dones = []

        self.minibatch_size = minibatch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.minibatch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.minibatch_size] for i in batch_start]

        return np.array(self.states), \
                np.array(self.actions), \
                np.array(self.probs), \
                np.array(self.vals), \
                np.array(self.entropies), \
                np.array(self.rewards), \
                np.array(self.dones), \
                batches
    
    def store_memory(self, state, prob, val, action, reward, done, entropy=None):
        self.states.append(state)
        self.probs.append(prob)
        self.vals.append(val)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.entropies.append(entropy)
    
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.entropies = []

class PPOAgent:
    def __init__(self, c1, c2, entropy_loss, 
                 minibatch_size,
                 policy_clip,
                 gamma, gae_lambda,
                 n_epochs,
                 adv_normalization,
                 actor, critic):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.c1 = c1
        self.c2 = c2
        self.entropy_loss = entropy_loss
        self.actor = actor
        self.critic = critic
        self.memory = PPOMemory(minibatch_size)
        self.adv_normalization = adv_normalization

    def remember(self, state, action, probs, vals, reward, done, entropy=None):
        self.memory.store_memory(state, probs, vals, action, reward, done, entropy)
    
    def save_checkpoint(self,path):
        T.save(self, path)

    def save_models(self,actorpath,criticpath):
        self.actor.save_checkpoint(actorpath)
        self.critic.save_checkpoint(criticpath)    

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        
        action, logprob, entropy = self.actor.getaction(state)
        value = self.critic(state)

    
        probs = T.squeeze(logprob).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        entropy = T.squeeze(entropy).item() if entropy is not None else None

        return action, probs, value, entropy

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, done_arr, entropy_arr, batches = \
                self.memory.generate_batches()
            
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)- 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1 - int(done_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            # Advantage normalization (once per epoch, before minibatches)
            if self.adv_normalization:
                advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-10)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = self.actor.get_log_prob(states, actions)
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + self.c1 * critic_loss
                if self.entropy_loss:
                    entropy_loss = -self.c2 * entropy_arr[batch].mean()
                    total_loss += entropy_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()  
