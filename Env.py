import gym
from gym import spaces
import itertools
import numpy as np
import torch
from tqdm import tqdm
import math

def sample_request_item(pos_items, pos_ratings, zipf_s=1.0):
    if len(pos_items) == 0:
        return -1
    pos_items = np.array(pos_items)
    pos_ratings = np.array(pos_ratings, dtype=np.float32)
    sorted_idx = np.argsort(-pos_ratings)
    sorted_items = pos_items[sorted_idx]
    ranks = np.arange(1, len(sorted_items) + 1)
    zipf_weights = 1.0 / (ranks ** zipf_s)
    probs = zipf_weights / zipf_weights.sum()
    chosen_idx = np.random.choice(len(sorted_items), p=probs)
    return int(sorted_items[chosen_idx])

class CarCachingEnv(gym.Env):
    """
    模拟路口中车辆、RSU 与缓存决策过程的环境。环境状态中除了包含车辆基本信息和缓存状态外，
    还加入了 LightGCN 在当前时刻对候选缓存集合的预测得分，供 RL 代理参考。
    """
    def __init__(self, args, crossroad, recommender, norm_adj, num_items,
                 test_user_ratings, cache_capacity=3, zipf_s=0.8, topk_candidate=20):
        super(CarCachingEnv, self).__init__()
        self.args = args
        self.crossroad = crossroad
        self.recommender = recommender    # 训练好的 LightGCN 模型
        self.norm_adj = norm_adj          # 归一化邻接矩阵（应位于 args.device 上）
        self.num_items = num_items
        self.test_user_ratings = test_user_ratings
        self.cache_capacity = cache_capacity
        self.zipf_s = zipf_s
        self.topk_candidate = topk_candidate

        # 初始时候候选集合和动作空间暂时为空，待 reset 时更新
        self.cache_candidate_set = None
        self.cache_candidate_scores = None  # 新增变量，用于存储 LightGCN 得分
        self.action_list = None

        # 状态空间：车辆平均位置(2) + 平均速度(2) + 平均带宽(1) + 当前缓存状态(cache_capacity)
        # + 当前候选集合的得分(topk_candidate)
        self.feature_dim = 2 + 2 + 1 + self.cache_capacity + self.topk_candidate
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_dim,), dtype=np.float32)
        self.action_dim_init = math.comb(self.topk_candidate, self.cache_capacity)
        self.action_space = spaces.Discrete(self.action_dim_init)

        self.current_cache = None  # 当前缓存配置
        self.hit_reward = 1.0
        self.miss_penalty = -1.0
        self.num_requests_per_vehicle = 1
        self.max_steps = args.max_steps
        self.current_step = 0

    def update_candidate_set(self):
        """
        利用 LightGCN 根据当前环境中车辆用户的预测，计算所有物品得分，
        并选取 topk_candidate 个物品作为候选缓存集合，同时保存对应的得分。
        """
        user_ids = [v.user_id for v in self.crossroad.vehicles]
        if len(user_ids) == 0:
            self.cache_candidate_set = np.arange(0, min(100, self.num_items))
            self.cache_candidate_scores = np.zeros(self.topk_candidate)
        else:
            user_tensor = torch.tensor(user_ids, dtype=torch.long).to(device=self.args.device)
            with torch.no_grad():
                user_embeds, item_embeds = self.recommender(self.norm_adj)
                # 若设备为 mps，则在 CPU 上进行索引操作
                if self.args.device == "mps":
                    user_tensor_cpu = user_tensor.cpu()
                    user_embeds_cpu = user_embeds.cpu()
                    avg_user_embed = user_embeds_cpu[user_tensor_cpu].mean(dim=0, keepdim=True)
                    avg_user_embed = avg_user_embed.to(self.args.device)
                else:
                    avg_user_embed = user_embeds[user_tensor].mean(dim=0, keepdim=True)
                scores = torch.matmul(avg_user_embed, item_embeds.t()).squeeze(0).cpu()
                topk = torch.topk(scores, self.topk_candidate)
                topk_items = topk.indices.numpy()
                topk_scores = topk.values.numpy()
                self.cache_candidate_set = topk_items
                self.cache_candidate_scores = topk_scores
        # 更新动作空间：候选集合中选择 cache_capacity 个物品的所有组合
        self.action_list = list(itertools.combinations(range(len(self.cache_candidate_set)), self.cache_capacity))
        self.num_actions = len(self.action_list)
        self.action_space = spaces.Discrete(self.num_actions)
        # 默认初始缓存选择第一个组合
        self.current_cache = self.action_list[0]

    def reset(self):
        self.crossroad.vehicles = []
        for i in range(self.args.num_vehicles):
            self.crossroad.generate_vehicle(user_id=i)
        self.update_candidate_set()
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        # 每一步更新候选集合
        if self.current_step % 2 == 0:
            self.update_candidate_set()
        self.current_cache = self.action_list[action]
        cache_files = [int(self.cache_candidate_set[idx]) for idx in self.current_cache]
        total_reward = 0.0
        hit_count = 0
        with torch.no_grad():
            user_embeds, item_embeds = self.recommender(self.norm_adj)
        hit = False
        for vehicle in tqdm(self.crossroad.vehicles, desc="Processing vehicles", leave=False):
            uid = vehicle.user_id
            for _ in range(self.num_requests_per_vehicle):
                if uid in self.test_user_ratings:
                    pos_items, pos_ratings = self.test_user_ratings[uid]
                else:
                    pos_items, pos_ratings = [], []
                requested_item = sample_request_item(pos_items, pos_ratings, self.zipf_s)
                if requested_item == -1:
                    continue

                distance = np.linalg.norm(vehicle.position - self.crossroad.rsu.position)
                normalized_distance = distance / self.crossroad.max_distance  # 归一化距离
                # 此处不考虑传输延迟，简单采用命中给正奖励，未命中给负奖励
                if requested_item in cache_files:
                    qoe_gain = 1.0 - normalized_distance
                    # 放大奖励信号
                    total_reward += (5 + qoe_gain) * 50 * hit_count  # 例如乘以 10
                    hit_count += 1
                else:
                    # 未命中，给予轻微惩罚
                    total_reward -= 20

        num_total = self.num_requests_per_vehicle * len(self.crossroad.vehicles)
        avg_reward = total_reward / num_total if num_total > 0 else 0.0

        self.crossroad.simulate_step(dt=self.args.cross_dt)
        self.current_step += 1
        next_state = self._get_state()
        done = (self.current_step >= self.max_steps)
        info = {
            "cache_hits": hit_count,
            "total_requests": num_total,
            "hit_rate": hit_count / num_total if num_total > 0 else 0.0
        }
        if self.current_step % 50 == 0:
            print(f"Step {self.current_step}, Avg Reward: {avg_reward:.4f}, "
                  f"Hit Rate: {info['hit_rate']:.4f}, Cache Hits: {hit_count}/{num_total}")
        return next_state, avg_reward, done, info

    def _get_state(self):
        if len(self.crossroad.vehicles) > 0:
            positions = np.array([v.position for v in self.crossroad.vehicles])
            speeds = np.array([v.speed for v in self.crossroad.vehicles])
            bandwidths = np.array([v.get_bandwidth(self.crossroad.rsu.position) for v in self.crossroad.vehicles])
            avg_position = positions.mean(axis=0)
            avg_speed = speeds.mean(axis=0)
            avg_bandwidth = np.array([bandwidths.mean()])
        else:
            avg_position = np.zeros(2)
            avg_speed = np.zeros(2)
            avg_bandwidth = np.zeros(1)
        # 当前缓存状态：候选集合中的物品ID归一化（除以 num_items）
        cache_state = np.array([self.cache_candidate_set[idx] / self.num_items for idx in self.current_cache])
        # 新增：加入当前 LightGCN 计算的候选得分（直接使用 self.cache_candidate_scores）
        candidate_scores = np.array(self.cache_candidate_scores)  # shape: (topk_candidate,)
        state = np.concatenate([avg_position, avg_speed, avg_bandwidth, cache_state, candidate_scores])
        return state.astype(np.float32)
