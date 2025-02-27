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
    def __init__(self, args, state_dim, action_dim, lr=0.0001, gamma=0.5,
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.05,
                 buffer_size=10000, batch_size=64):
        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.device = self.args.device
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


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random

# 定义 Actor 网络（输出每个动作的 logits）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # 输出 logits，交由 Categorical 计算概率
        )
    def forward(self, x):
        return self.net(x)

# 定义 Critic 网络（输出状态价值）
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# PPO Agent 实现
class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99,
                 eps_clip=0.2, K_epochs=4, entropy_coef=0.01, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip      # PPO 剪切范围
        self.K_epochs = K_epochs      # 每次更新的轮数
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)

        # 初始化 Actor 与 Critic 网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        # 使用同一个优化器更新 Actor 与 Critic 参数
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                    lr=actor_lr)
        # 用于存储当前 episode 的经验（state, action, logprob, reward, done, next_state）
        self.memory = []

    def select_action(self, state):
        """
        根据当前 state 采样动作，并返回动作以及该动作的 log 概率
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.actor(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), logprob.item()

    def store_transition(self, state, action, logprob, reward, done, next_state):
        self.memory.append((state, action, logprob, reward, done, next_state))

    def clear_memory(self):
        self.memory = []

    def update(self):
        # 将 memory 转换为 tensor
        states = torch.FloatTensor([trans[0] for trans in self.memory]).to(self.device)
        actions = torch.LongTensor([trans[1] for trans in self.memory]).to(self.device)
        old_logprobs = torch.FloatTensor([trans[2] for trans in self.memory]).to(self.device)
        rewards = [trans[3] for trans in self.memory]
        dones = [trans[4] for trans in self.memory]
        next_states = torch.FloatTensor([trans[5] for trans in self.memory]).to(self.device)

        # 计算折扣回报（简单累积回报，没有使用时序归一化，可根据需要调整）
        discounted_rewards = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        # 标准化回报
        if discounted_rewards.std() > 1e-5:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # 多个 epoch 更新 PPO
        for _ in range(self.K_epochs):
            logits = self.actor(states)
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Critic 评估状态价值
            state_values = self.critic(states).squeeze()
            advantages = discounted_rewards - state_values.detach()

            # PPO 目标函数，采用 clip 策略
            ratios = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
            critic_loss = F.mse_loss(state_values, discounted_rewards)
            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.clear_memory()
        return loss.item()


# Soft Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)  # Q-values for each action


# Actor 网络: 输出策略分布
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=-1)  # 动作概率


# DSAC Agent
class DSACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=10000,
                 batch_size=64, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # 温度参数，控制探索程度
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.device = device

        # 创建 Actor 和 Critic 网络
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.q_net1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q_net2 = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net1 = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net2 = QNetwork(state_dim, action_dim).to(self.device)

        # 复制目标网络参数
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=lr)

    def select_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy_net(state_tensor)
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
        return action

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

        # 计算目标 Q 值
        with torch.no_grad():
            next_probs = self.policy_net(next_states)
            next_dist = torch.distributions.Categorical(next_probs)
            next_actions = next_dist.sample()
            next_log_probs = next_dist.log_prob(next_actions).unsqueeze(1)
            next_q1 = self.target_q_net1(next_states).gather(1, next_actions.unsqueeze(1))
            next_q2 = self.target_q_net2(next_states).gather(1, next_actions.unsqueeze(1))
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * next_q

        # 更新 Q 网络
        q1_values = self.q_net1(states).gather(1, actions)
        q2_values = self.q_net2(states).gather(1, actions)
        q_loss1 = F.mse_loss(q1_values, target_q)
        q_loss2 = F.mse_loss(q2_values, target_q)

        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        self.q_optimizer2.step()

        # 更新策略网络
        probs = self.policy_net(states)
        dist = torch.distributions.Categorical(probs)
        sampled_actions = dist.sample()
        log_probs = dist.log_prob(sampled_actions).unsqueeze(1)

        q1_new = self.q_net1(states).gather(1, sampled_actions.unsqueeze(1))
        q2_new = self.q_net2(states).gather(1, sampled_actions.unsqueeze(1))
        min_q = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_probs - min_q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 软更新目标 Q 网络
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

