import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from collections import deque
import itertools
from tqdm import tqdm
import os


# 假设前面已有的 Vehicle、RSU、Crossroad、CarCachingEnv、sample_request_item 等已正确定义和导入

#############################################
# 传统缓存策略仿真函数：LRU 与 LFU
#############################################
def simulate_traditional_policy(env, policy, num_episodes):
    """
    对传统缓存策略进行仿真：
      - 每个 episode 中首先调用 env.reset() 重置车辆状态，
      - 初始化 RSU 缓存为空，
      - 在每个时间步中，遍历所有车辆采样请求，判断缓存命中（命中则更新缓存状态；未命中则插入缓存——若已满则根据策略进行淘汰）。
    支持的策略：
      "LRU" - 最近最少使用：命中后将该项移动到队尾，未命中时若缓存满则淘汰队首项；
      "LFU" - 最不经常使用：每个缓存项记录使用频次，命中后频次加 1；未命中时若缓存满则淘汰使用频次最小的项。
    返回每个 episode 的命中率列表。
    """
    hit_ratios = []
    max_steps = env.max_steps  # 与 RL 仿真保持一致

    for ep in range(num_episodes):
        env.reset()
        # 初始化缓存状态及（对于 LFU）使用频率记录
        if policy == "LFU":
            cache_state = []  # 缓存中的物品列表
            frequency = {}  # 记录每个缓存项的使用频次
        else:  # 默认为 LRU 策略
            cache_state = []
        total_hits = 0
        total_requests = 0

        for step in range(max_steps):
            # 遍历所有车辆，模拟请求
            for vehicle in env.crossroad.vehicles:
                uid = vehicle.user_id
                if uid in env.test_user_ratings:
                    pos_items, pos_ratings = env.test_user_ratings[uid]
                else:
                    pos_items, pos_ratings = [], []
                requested_item = sample_request_item(pos_items, pos_ratings, env.zipf_s)
                if requested_item == -1:
                    continue
                total_requests += 1

                # 判断是否命中缓存
                if requested_item in cache_state:
                    total_hits += 1
                    if policy == "LRU":
                        # 命中后将该项移至末尾表示最近使用
                        cache_state.remove(requested_item)
                        cache_state.append(requested_item)
                    elif policy == "LFU":
                        frequency[requested_item] += 1
                else:
                    # 未命中：插入缓存，若缓存满则淘汰
                    if policy == "LRU":
                        if len(cache_state) < env.cache_capacity:
                            cache_state.append(requested_item)
                        else:
                            # 淘汰最久未使用的（队首）
                            cache_state.pop(0)
                            cache_state.append(requested_item)
                    elif policy == "LFU":
                        if len(cache_state) < env.cache_capacity:
                            cache_state.append(requested_item)
                            frequency[requested_item] = 1
                        else:
                            # 淘汰使用频次最小的项
                            lfu_item = min(cache_state, key=lambda x: frequency.get(x, 0))
                            cache_state.remove(lfu_item)
                            del frequency[lfu_item]
                            cache_state.append(requested_item)
                            frequency[requested_item] = 1
            # 模拟下一步车辆运动、生成与删除
            env.crossroad.simulate_step(dt=1.0)
        hit_ratio = total_hits / total_requests if total_requests > 0 else 0
        hit_ratios.append(hit_ratio)
        print(f"传统策略 {policy} - Episode {ep + 1}/{num_episodes}, Hit Ratio: {hit_ratio:.4f}")
    return hit_ratios


#############################################
# 主流程（对比 RL 与传统缓存策略：LRU 与 LFU）
#############################################
def main():
    # 加载配置、数据与模型（假设相关函数均已定义）
    from config import get_config
    args = get_config()
    device = args.device

    # 加载 ml-1m 数据、构建用户-电影矩阵、归一化邻接矩阵、测试集等
    from autoencoder import load_ml1m, build_user_item_matrix, create_norm_adj_matrix
    filepath = "ml-1m/ratings.dat"
    df = load_ml1m(filepath)
    num_users = df['UserID'].max() + 1
    num_items = df['MovieID'].max() + 1
    print("用户总数:", num_users, "电影总数:", num_items)
    train_np, test_np = build_user_item_matrix(df, num_users, num_items, split_ratio=0.8)
    norm_adj = create_norm_adj_matrix(train_np)

    # 构造测试用户评分字典
    def construct_test_user_ratings(test_matrix):
        test_user_ratings = {}
        num_users = test_matrix.shape[0]
        for u in range(num_users):
            pos_items = np.where(test_matrix[u] > 0)[0]
            pos_ratings = test_matrix[u][pos_items]
            test_user_ratings[u] = (pos_items.tolist(), pos_ratings.tolist())
        return test_user_ratings

    test_user_ratings = construct_test_user_ratings(test_np)

    # 初始化 Crossroad（区域尺寸 100x100）
    from Vehicle import Crossroad  # 假设已有定义
    crossroad = Crossroad(width=100, height=100)

    # 加载训练好的 LightGCN 模型
    from autoencoder import LightGCN
    embedding_dim = args.embedding_dim
    num_layers = args.num_layers
    recommender = LightGCN(num_users, num_items, embedding_dim, num_layers, dropout=args.dropout)
    if torch.cuda.is_available() or args.device != 'cpu':
        recommender.to(device)
    map_location = None if torch.cuda.is_available() else 'cpu'
    if os.path.exists('lightgcn_model.pth'):
        recommender.load_state_dict(torch.load('lightgcn_model.pth', map_location=map_location))
        recommender.eval()
        print("LightGCN 模型已加载。")
    else:
        print("请先训练 LightGCN 模型。")
        return

    # 初始化车联网缓存环境（RL 环境）
    from Env import CarCachingEnv  # 假设已有定义
    env_rl = CarCachingEnv(args, crossroad, recommender, norm_adj, num_items, test_user_ratings,
                           cache_capacity=args.cache_capacity, zipf_s=args.zipf_s, topk_candidate=20)

    num_episodes = 200
    target_update_freq = 10

    #############################################
    # RL 代理训练（以 DSAC 为例，可根据 args.agent 选择）
    #############################################
    if args.agent == 'ppo':
        from Agent import PPOAgent
        agent = PPOAgent(state_dim=env_rl.observation_space.shape[0], action_dim=env_rl.action_space.n,
                         device=args.device)
    elif args.agent == 'dqn':
        from Agent import DQNAgent
        agent = DQNAgent(args, state_dim=env_rl.observation_space.shape[0], action_dim=env_rl.action_space.n)
    elif args.agent == 'dsac':
        from Agent import DSACAgent
        agent = DSACAgent(state_dim=env_rl.observation_space.shape[0], action_dim=env_rl.action_space.n,
                          device=args.device)
    else:
        print("未知的代理类型")
        return

    rl_hit_ratios = []
    for episode in range(num_episodes):
        state = env_rl.reset()
        done = False
        total_reward = 0.0
        episode_hit_count = 0
        episode_requests = 0
        while not done:
            if args.agent == 'ppo':
                action, log_prob = agent.select_action(state)
                next_state, reward, done, info = env_rl.step(action)
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
        print(f"RL Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.4f}, Hit Ratio: {hit_ratio:.4f}")

    #############################################
    # 传统缓存策略仿真：LRU 与 LFU（使用新的环境实例以保证初始化一致）
    #############################################
    crossroad_lru = Crossroad(width=100, height=100)
    env_lru = CarCachingEnv(args, crossroad_lru, recommender, norm_adj, num_items, test_user_ratings,
                            cache_capacity=args.cache_capacity, zipf_s=args.zipf_s, topk_candidate=20)
    lru_hit_ratios = simulate_traditional_policy(env_lru, policy="LRU", num_episodes=num_episodes)

    crossroad_lfu = Crossroad(width=100, height=100)
    env_lfu = CarCachingEnv(args, crossroad_lfu, recommender, norm_adj, num_items, test_user_ratings,
                            cache_capacity=args.cache_capacity, zipf_s=args.zipf_s, topk_candidate=20)
    lfu_hit_ratios = simulate_traditional_policy(env_lfu, policy="LFU", num_episodes=num_episodes)

    #############################################
    # 绘制对比图：RL vs LRU vs LFU
    #############################################
    plt.figure(figsize=(8, 5))
    plt.plot(rl_hit_ratios, label="RL Agent")
    plt.plot(lru_hit_ratios, label="LRU")
    plt.plot(lfu_hit_ratios, label="LFU")
    plt.xlabel("Episode")
    plt.ylabel("Hit Ratio")
    plt.title("缓存策略对比 (Hit Ratio over Episodes)")
    plt.legend()
    plt.savefig("caching_policy_comparison.png")
    plt.show()


if __name__ == '__main__':
    main()
