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
from Vehicle import Crossroad
from Vehicle import RSU
from Vehicle import Vehicle
from dataloader import MovieLensDataLoader
from Env import sample_request_item
from Env import CarCachingEnv
from autoencoder import LightGCN
from Agent import DQNAgent
import matplotlib.pyplot as plt

from autoencoder import load_ml1m, build_user_item_matrix, create_norm_adj_matrix
def construct_test_user_ratings(test_matrix):
    """
    构造一个字典：user -> (pos_items, pos_ratings)
    """
    test_user_ratings = {}
    num_users = test_matrix.shape[0]
    for u in range(num_users):
        pos_items = np.where(test_matrix[u] > 0)[0]
        pos_ratings = test_matrix[u][pos_items]
        test_user_ratings[u] = (pos_items.tolist(), pos_ratings.tolist())
    return test_user_ratings

#####################################
# 主流程
#####################################
def main():
    # 检查GPU
    if not torch.cuda.is_available():
        print("GPU不可用，请检查CUDA配置。")
        return

    # 加载 ml-1m 数据
    filepath = "ml-1m/ratings.dat"
    df = load_ml1m(filepath)
    num_users = df['UserID'].max() + 1
    num_items = df['MovieID'].max() + 1
    print("用户总数:", num_users, "电影总数:", num_items)

    train_np, test_np = build_user_item_matrix(df, num_users, num_items, split_ratio=0.8)
    norm_adj = create_norm_adj_matrix(train_np)
    test_user_ratings = construct_test_user_ratings(test_np)

    # 初始化 Crossroad
    # 这里简单设定区域尺寸为 100x100
    from Vehicle import Crossroad  # 假设 Vehicle.py 中包含 Crossroad 类
    crossroad = Crossroad(width=100, height=100)

    # 加载训练好的 LightGCN 模型
    embedding_dim = 64
    num_layers = 3
    recommender = LightGCN(num_users, num_items, embedding_dim, num_layers, dropout=0.1).cuda()
    if os.path.exists('lightgcn_model.pth'):
        recommender.load_state_dict(torch.load('lightgcn_model.pth'))
        recommender.eval()
        print("LightGCN 模型已加载。")
    else:
        print("请先训练 LightGCN 模型。")
        return

    # 初始化车联网缓存环境
    env = CarCachingEnv(crossroad, recommender, norm_adj, num_items, test_user_ratings,
                          cache_capacity=20, zipf_s=1.0)

    # 初始化 DQN 代理
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    num_episodes = 200
    target_update_freq = 10
    hit_ratios = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        episode_hit_ratio = 0.0
        episode_requests = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
            episode_hit_ratio += info['cache_hits']
            episode_requests += info['total_requests']
        if (episode + 1) % target_update_freq == 0:
            agent.update_target()
        hit_ratios.append(episode_hit_ratio / episode_requests)
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
    plot_hit_ratio(hit_ratios)

    # 保存训练好的代理模型
def plot_hit_ratio(hit_ratio_list):
    plt.plot(hit_ratio_list)
    plt.xlabel("Episode")
    plt.ylabel("Hit Ratio")
    plt.title("Hit Ratio over Episodes")
    plt.savefig("hit_ratio.png")
    plt.show()
if __name__ == '__main__':
    main()
