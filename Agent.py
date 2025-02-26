import os
import random
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from gym import spaces
import gym

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.5,
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.05,
                 buffer_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, target)
        print(f"Loss: {loss.item()}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# Actor 网络：输入 state，输出各动作的概率分布
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        probs = F.softmax(logits, dim=-1)
        return probs


# Critic 网络：输入 state，输出状态价值
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.out(x)
        return value


# A2C Agent：同时维护 actor 与 critic，提供选择动作和更新的方法
class A2CAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, entropy_coef=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        """
        根据当前状态选择动作，并返回动作、该动作的对数概率和熵（用于训练时鼓励探索）。
        state: numpy 数组 (state_dim,)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # shape: (1, state_dim)
        probs = self.actor(state_tensor)  # shape: (1, action_dim)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def update(self, transitions):
        """
        根据一批 transition 更新 actor 与 critic 网络。
        transitions: list of (state, action, reward, next_state, done, log_prob, entropy)
        """
        # 将各部分转换为 tensor
        states, actions, rewards, next_states, dones, log_probs, entropies = zip(*transitions)
        states = torch.FloatTensor(states).to(self.device)  # shape: (batch, state_dim)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # shape: (batch, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # shape: (batch, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)  # shape: (batch, state_dim)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)  # shape: (batch, 1)
        log_probs = torch.stack(log_probs)  # shape: (batch,)
        entropies = torch.stack(entropies)  # shape: (batch,)

        # 计算当前状态的价值
        values = self.critic(states)  # shape: (batch, 1)
        # 计算下一个状态的价值
        next_values = self.critic(next_states)  # shape: (batch, 1)
        # 计算 target：r + gamma * V(next_state) * (1 - done)
        targets = rewards + self.gamma * next_values * (1 - dones)
        # 计算 advantage
        advantages = targets - values

        # Actor 损失：使 log_prob * advantage 最大化，同时加上熵正则化
        actor_loss = - (log_probs * advantages.detach()).mean() - self.entropy_coef * entropies.mean()
        # Critic 损失：均方误差
        critic_loss = F.mse_loss(values, targets.detach())

        loss = actor_loss + critic_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item()
