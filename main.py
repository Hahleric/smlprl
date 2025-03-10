import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from collections import deque
import itertools
from tqdm import tqdm
import os
import datetime
from Env import sample_request_item
from config import get_config
# 假设前面已有的 Vehicle、RSU、Crossroad、CarCachingEnv、sample_request_item 等已正确定义和导入
args = get_config()
#############################################
# 传统缓存策略仿真函数：LRU 与 LFU
#############################################
#############################################
# 传统缓存策略仿真函数：LRU、LFU、FIFO、MRU、Random、2Q
#############################################
def simulate_traditional_policy(env, policy, num_episodes):
    """
    传统缓存策略仿真：
      - LRU: 最近最少使用，淘汰最久未访问的项
      - LFU: 最不经常使用，淘汰访问次数最少的项
      - FIFO: 先进先出，淘汰最早进入缓存的项
      - MRU: 最近最常使用，淘汰最近使用的项
      - Random: 随机淘汰缓存中的某一项
      - 2Q: 两级队列缓存，先用 FIFO 存储，访问多次后进入 LRU
    """
    hit_ratios = []
    max_steps = env.max_steps  # 与 RL 仿真保持一致

    for ep in range(num_episodes):
        env.reset()
        # 初始化缓存状态及访问记录
        cache_state = []
        frequency = {}  # LFU 访问频率
        queue_2q = []  # 2Q FIFO 队列
        queue_size = env.cache_capacity // 2  # 2Q: FIFO 部分大小
        total_hits = 0
        total_requests = 0

        for step in range(max_steps):
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
                        cache_state.remove(requested_item)
                        cache_state.append(requested_item)
                    elif policy == "LFU":
                        frequency[requested_item] += 1
                    elif policy == "MRU":
                        # MRU 淘汰最近访问的项目，因此每次命中后删除该项
                        cache_state.remove(requested_item)
                    elif policy == "2Q":
                        # 如果在 LRU 阶段被访问，则移动到队列末尾
                        if requested_item in queue_2q:
                            queue_2q.remove(requested_item)
                            queue_2q.append(requested_item)
                else:
                    # 处理未命中情况
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
                                queue_2q.pop(0)  # FIFO 淘汰
                            cache_state.pop(0)  # LRU 淘汰
                        cache_state.append(requested_item)
                        if policy == "LFU":
                            frequency[requested_item] = 1
                        if policy == "2Q":
                            queue_2q.append(requested_item)

            env.crossroad.simulate_step(dt=args.cross_dt)

        hit_ratio = total_hits / total_requests if total_requests > 0 else 0
        hit_ratios.append(hit_ratio)
        print(f"传统策略 {policy} - Episode {ep + 1}/{num_episodes}, Hit Ratio: {hit_ratio:.4f}")

    return hit_ratios



#############################################
# 主流程（对比 RL 与传统缓存策略：LRU 与 LFU）
#############################################
def main():
    # 加载配置、数据与模型（假设相关函数均已定义）

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

    num_episodes = args.episodes
    target_update_freq = 10

    #############################################
    # RL 代理训练（以 DSAC 为例，可根据 args.agent 选择）
    #############################################
    if args.agent == 'ppo':
        from Agent import PPOAgent
        agent = PPOAgent(state_dim=env_rl.observation_space.shape[0], action_dim=env_rl.action_space.n,gamma=args.gamma,
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
        rewarded = False
        total_reward = 0.0
        episode_hit_count = 0
        episode_requests = 0
        while not done:
            if args.agent == 'ppo':
                action, log_prob = agent.select_action(state)
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
        print(f"RL Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.4f}, Hit Ratio: {hit_ratio:.4f}")

    #############################################
    # 传统缓存策略仿真：LRU、LFU、FIFO、MRU、Random、2Q
    #############################################
    policies = ["LRU", "FIFO", "MRU", "Random", "2Q"]
    traditional_results = {}

    for policy in policies:
        crossroad_trad = Crossroad(width=100, height=100)
        env_trad = CarCachingEnv(args, crossroad_trad, recommender, norm_adj, num_items, test_user_ratings,
                                 cache_capacity=args.cache_capacity, zipf_s=args.zipf_s, topk_candidate=20)
        traditional_results[policy] = simulate_traditional_policy(env_trad, policy=policy, num_episodes=num_episodes)

    #############################################
    # 绘制对比图：RL vs 传统策略（LRU、LFU、FIFO、MRU、Random、2Q）
    #############################################
    plt.figure(figsize=(10, 6))
    plt.plot(rl_hit_ratios, label="RL Agent", linewidth=2)

    for policy, hit_ratios in traditional_results.items():
        plt.plot(hit_ratios, label=policy)

    plt.xlabel("Episode")
    plt.ylabel("Hit Ratio")
    plt.title("policies comparison (Hit Ratio over Episodes)")
    plt.legend()
    plt.savefig("caching_policy_comparison_" + str(datetime.datetime.now()) + ".png")
    plt.show()



if __name__ == '__main__':
    main()
