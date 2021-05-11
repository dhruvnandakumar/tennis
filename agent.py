import numpy as np
import random
import copy
import os
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, state_size, action_size, seed, hparams, identity):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed) 
        self.hparams = hparams
        self.identity = identity
        
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.hparams["lr_actor"])
        
        for target_param, source_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(source_param.data)
        
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.hparams["lr_critic"], weight_decay=self.hparams["weight_decay"])
        
        for target_param, source_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(source_param.data)
        
        
        #Controller will handle shared memory
        self.memory = ReplayBuffer(action_size, self.hparams["buffer_size"], self.hparams["batch_size"], seed)
        self.noise = OUNoise(action_size, seed)
       
        
    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        
        #Controller will handle concatenating the actions from each agent
        if not torch.is_tensor(states):
            states = torch.from_numpy(states).float().to(device)
            
        self.actor_local.eval()
        with torch.no_grad():
                action = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * self.hparams['epsilon']
        return np.clip(action, -1, 1)
    
    # Handle step in controller
    def step(self, states, actions, rewards, next_states, dones, ep):
        self.memory.add(states, actions, rewards, next_states, dones)
        
        if len(self.memory) > self.hparams["batch_size"] and ep % 5 == 0 and ep > 100:
            for _ in range(4):
                experiences = self.memory.sample()
                self.learn(experiences)
                    
    
    def learn(self, experiences):
        
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.hparams["gamma"] * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
        self.hparams['epsilon'] *= self.hparams['epsilon_decay']
    
    def reset(self):
        self.noise.reset()    
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.hparams["tau"]*local_param.data + (1.0-self.hparams["tau"])*target_param.data)
            
    def print_models(self):
        print("Agent ", str(self.identity), " ", self.actor_local)
        print("Agent ", str(self.identity), " ", self.critic_local)
        
    def save_models(self):
        torch.save(self.actor_local.state_dict(), str(self.identity) + "_actor_weights.pth")
        torch.save(self.critic_local.state_dict(), str(self.identity) + "_critic_weights.pth")
        
    


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma        
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state        
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state