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
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from VideoCallback import VideoRecordingCallback

from config import get_config
args = get_config()

#############################################
# 加载数据、构建用户-电影矩阵、归一化邻接矩阵、测试集等（假设相关函数已定义）
from autoencoder import load_ml1m, build_user_item_matrix, create_norm_adj_matrix
filepath = args.ratings_path
df = load_ml1m(filepath)
num_users = df['UserID'].max() + 1
num_items = df['MovieID'].max() + 1
print("用户总数:", num_users, "电影总数:", num_items)
train_np, test_np = build_user_item_matrix(df, num_users, num_items, split_ratio=0.8)
norm_adj = create_norm_adj_matrix(train_np)

def construct_test_user_ratings(test_matrix):
    test_user_ratings = {}
    num_users = test_matrix.shape[0]
    for u in range(num_users):
        pos_items = np.where(test_matrix[u] > 0)[0]
        pos_ratings = test_matrix[u][pos_items]
        test_user_ratings[u] = (pos_items.tolist(), pos_ratings.tolist())
    return test_user_ratings

test_user_ratings = construct_test_user_ratings(test_np)

#############################################
# 初始化 Crossroad（传入 base_request_frequency）
from Vehicle import Crossroad
crossroad = Crossroad(width=100, height=100, base_request_frequency=args.base_request_frequency)

#############################################
# 加载训练好的 LightGCN 模型
from autoencoder import LightGCN
embedding_dim = args.embedding_dim
num_layers = args.num_layers
recommender = LightGCN(num_users, num_items, embedding_dim, num_layers, dropout=args.dropout)
if torch.cuda.is_available() or args.device != 'cpu':
    recommender.to(args.device)
map_location = None if torch.cuda.is_available() else 'cpu'
if os.path.exists('lightgcn_model.pth'):
    recommender.load_state_dict(torch.load('lightgcn_model.pth', map_location=map_location))
    recommender.eval()
    print("LightGCN 模型已加载。")
else:
    print("请先训练 LightGCN 模型。")
    exit()

#############################################
# 初始化车联网缓存环境（RL 环境）
from Env import CarCachingEnv, sample_request_item, GNNCarCachingEnv

# 使用 RL 环境时，使用 args.rl_zipf_s
if args.use_gnn:
    env_rl = GNNCarCachingEnv(args, crossroad, recommender, norm_adj, num_items, test_user_ratings,
                              cache_capacity=args.cache_capacity, zipf_s=args.rl_zipf_s, topk_candidate=20,
                              use_recommendation_boost=True)
else:
    env_rl = CarCachingEnv(args, crossroad, recommender, norm_adj, num_items, test_user_ratings,
                           cache_capacity=args.cache_capacity, zipf_s=args.rl_zipf_s, topk_candidate=20,
                           use_recommendation_boost=True)
num_episodes = args.episodes
target_update_freq = 10

#############################################
# RL 代理训练（以 PPO 为例）
if args.use_sbl:
    from stable_baselines3 import PPO
    if args.use_gnn:
        from gnn_agent import TorchGeoGNNPPOPolicy

        ppo_model = PPO(TorchGeoGNNPPOPolicy, env_rl, verbose=1, device=args.sbl_device, n_steps=args.max_steps,
                        policy_kwargs=dict(conv_type=args.gnn_conv_type))
    else:
        ppo_model = PPO("MlpPolicy", env_rl, verbose=1, device=args.sbl_device, n_steps=args.max_steps)

    if not args.use_saved_rl:
        ppo_model.learn(total_timesteps=20480, progress_bar=True)
        ppo_model.save("ppo_model" + args.gnn_conv_type)
    loaded_model = PPO.load("ppo_model" + args.gnn_conv_type +".zip")
    obs = env_rl.reset()
    rl_hit_ratios = []
    for _ in tqdm(range(args.episodes)):
        hits = 0
        requests = 0
        for _ in tqdm(range(args.testing_step)):
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, info = env_rl.step(action)
            hits += info['cache_hits']
            requests += info['total_requests']
            if done:
                obs = env_rl.reset()
        rl_hit_ratios.append(hits / requests if requests > 0 else 0)
else:
        if args.agent == 'ppo':
            from agent import PPOAgent

            agent = PPOAgent(state_dim=env_rl.observation_space.shape[0], action_dim=env_rl.action_space.n,
                             gamma=args.gamma,
                             device=args.device)
        elif args.agent == 'dqn':
            from agent import DQNAgent

            agent = DQNAgent(args, state_dim=env_rl.observation_space.shape[0], action_dim=env_rl.action_space.n)
        elif args.agent == 'dsac':
            from agent import DSACAgent

            agent = DSACAgent(state_dim=env_rl.observation_space.shape[0], action_dim=env_rl.action_space.n,
                              device=args.device)
        else:
            print("未知的代理类型")
            exit(1)

        rl_hit_ratios = []
        for episode in range(num_episodes):
            state = env_rl.reset()
            done = False
            rewarded = False
            total_reward = 0.0
            episode_hit_count = 0
            episode_requests = 0
            hidden_state = None
            while not done:
                if args.agent == 'ppo':
                    action, log_prob, new_hidden_state = agent.select_action(state, hidden_state)
                    hidden_state = new_hidden_state
                    next_state, reward, done, info = env_rl.step(action)
                    if not rewarded and episode_hit_count > args.hit_threshold:
                        rewarded = True
                        reward += 1000
                    agent.store_transition(state, action, log_prob, reward, done, next_state)
                elif args.agent in ['dqn', 'dsac']:
                    action = agent.select_action(state)
                    next_state, reward, done, info = env_rl.step(action)
                    agent.store_transition(state, action, reward, next_state, done)
                    agent.train_step()
                state = next_state
                total_reward += reward
                episode_hit_count += info['cache_hits']
                episode_requests += info['total_requests']
            if args.agent == 'ppo':
                agent.update()
            elif args.agent == 'dqn' and (episode + 1) % target_update_freq == 0:
                agent.update_target()
            hit_ratio = episode_hit_count / episode_requests if episode_requests > 0 else 0
            rl_hit_ratios.append(hit_ratio)
            print(
                f"RL Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.4f}, Hit Ratio: {hit_ratio:.4f}")

#############################################
# 传统缓存策略仿真：LRU、FIFO、MRU、Random、2Q
def simulate_traditional_policy(env, policy, num_episodes):
    """
    传统缓存策略仿真，同时模拟用户行为：
      - 每辆车按照其 request_frequency 发起多次请求；
      - 当命中缓存时增加其 request_frequency（request_frequency_increment）；
      - 当未命中时衰减其 request_frequency（request_frequency_decay，不能低于 base_request_frequency）。
    如果启用了推荐提升（use_recommendation_boost），则对属于 LightGCN 推荐候选集合的物品的评分进行提升。
    """
    hit_ratios = []
    max_steps = args.testing_step

    for ep in range(num_episodes):
        env.reset()
        cache_state = []  # 用于保存当前缓存中的物品（对于非 LFU 策略）
        frequency = {}  # 用于 LFU 策略记录各物品访问频率
        queue_2q = []  # 用于 2Q 策略
        queue_size = env.cache_capacity // 2
        total_hits = 0
        total_requests = 0

        for step in range(max_steps):
            # 对每辆车，按照其动态 request_frequency 发起请求
            for vehicle in env.crossroad.vehicles:
                num_requests = int(vehicle.request_frequency)
                for _ in range(num_requests):
                    uid = vehicle.user_id
                    if uid in env.test_user_ratings:
                        pos_items, pos_ratings = env.test_user_ratings[uid]
                    else:
                        pos_items, pos_ratings = [], []

                    # 如果启用了推荐提升，则对候选集合中的物品提升其评分
                    if args.recommendation_boost_trad and (env.cache_candidate_set is not None):
                        boost = env.args.recommendation_boost
                        pos_ratings = [r * boost if item in env.cache_candidate_set else r
                                       for item, r in zip(pos_items, pos_ratings)]

                    requested_item = sample_request_item(pos_items, pos_ratings, env.zipf_s)
                    if requested_item == -1:
                        continue
                    total_requests += 1

                    # 判断缓存命中
                    if requested_item in cache_state:
                        total_hits += 1
                        # 更新传统策略缓存逻辑
                        if policy == "LRU":
                            cache_state.remove(requested_item)
                            cache_state.append(requested_item)
                        elif policy == "LFU":
                            frequency[requested_item] += 1
                        elif policy == "MRU":
                            cache_state.remove(requested_item)
                        elif policy == "2Q":
                            if requested_item in queue_2q:
                                queue_2q.remove(requested_item)
                                queue_2q.append(requested_item)
                        # 增加请求频率
                        vehicle.request_frequency = min(vehicle.request_frequency + env.args.request_frequency_increment,
                                                        env.args.max_request_frequency)
                    else:
                        # 未命中，更新缓存状态
                        if len(cache_state) < env.cache_capacity:
                            cache_state.append(requested_item)
                            if policy == "LFU":
                                frequency[requested_item] = 1
                            if policy == "2Q":
                                queue_2q.append(requested_item)
                        else:
                            if policy == "LRU":
                                cache_state.pop(0)
                            elif policy == "LFU":
                                lfu_item = min(cache_state, key=lambda x: frequency.get(x, 0))
                                cache_state.remove(lfu_item)
                                del frequency[lfu_item]
                            elif policy == "FIFO":
                                cache_state.pop(0)
                            elif policy == "MRU":
                                cache_state.pop(-1)
                            elif policy == "Random":
                                cache_state.pop(random.randint(0, len(cache_state) - 1))
                            elif policy == "2Q":
                                if len(queue_2q) > queue_size:
                                    queue_2q.pop(0)
                                cache_state.pop(0)
                            cache_state.append(requested_item)
                            if policy == "LFU":
                                frequency[requested_item] = 1
                            if policy == "2Q":
                                queue_2q.append(requested_item)
                        # 衰减请求频率，但不低于初始值
                        vehicle.request_frequency = max(vehicle.request_frequency - env.args.request_frequency_decay,
                                                        env.args.base_request_frequency)
            # 模拟车辆更新（例如位置、生成新车辆等）
            env.crossroad.simulate_step(dt=env.args.cross_dt)
            if step % 5 == 0:
                env.update_candidate_set()
        hit_ratio = total_hits / total_requests if total_requests > 0 else 0
        hit_ratios.append(hit_ratio)
        print(f"传统策略 {policy} - Episode {ep + 1}/{num_episodes}, Hit Ratio: {hit_ratio:.4f}")

    return hit_ratios


policies = ["LRU", "FIFO", "MRU", "Random", "2Q"]
traditional_results = {}

for policy in policies:
    from Vehicle import Crossroad

    crossroad_trad = Crossroad(width=100, height=100, base_request_frequency=args.base_request_frequency)
    env_trad = CarCachingEnv(args, crossroad_trad, recommender, norm_adj, num_items, test_user_ratings,
                             cache_capacity=args.cache_capacity, zipf_s=args.trad_zipf_s, topk_candidate=20,
                             use_recommendation_boost=False)
    traditional_results[policy] = simulate_traditional_policy(env_trad, policy=policy, num_episodes=num_episodes)

#############################################
# 绘制对比图：RL vs 传统策略
plt.figure(figsize=(10, 6))
plt.plot(rl_hit_ratios, label="RL Agent", linewidth=2)
for policy, hit_ratios in traditional_results.items():
    plt.plot(hit_ratios, label=policy)
plt.xlabel("Episode")
plt.ylabel("Hit Ratio")
plt.title("Caching Policy Comparison (Hit Ratio over Episodes)")
plt.legend()
plt.savefig("caching_policy_comparison_" + str(datetime.datetime.now()) + ".png")
plt.show()
