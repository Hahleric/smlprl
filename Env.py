from typing import Optional, Union, List
import gym
from gym import spaces
import itertools
import numpy as np
import torch
from gym.core import RenderFrame
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
    模拟路口中车辆、RSU 与缓存决策过程的环境。
    状态包括车辆基本信息、缓存状态、候选集合得分以及新增的平均延迟信息。
    """
    def __init__(self, args, crossroad, recommender, norm_adj, num_items,
                 test_user_ratings, cache_capacity=3, zipf_s=1.0, topk_candidate=20,
                 use_recommendation_boost=True):
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
        self.use_recommendation_boost = use_recommendation_boost

        # 状态空间：车辆平均位置(2) + 平均速度(2) + 平均带宽(1) + 当前缓存状态(cache_capacity)
        # + 当前候选集合的得分(topk_candidate) + 平均延迟(1)
        self.feature_dim = 2 + 2 + 1 + self.cache_capacity + self.topk_candidate + 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.feature_dim,), dtype=np.float32)
        self.action_dim_init = math.comb(self.topk_candidate, self.cache_capacity)
        self.action_space = spaces.Discrete(self.action_dim_init)

        self.current_cache = None  # 当前缓存配置（记录候选集合索引组合）
        self.cache_candidate_set = None
        self.cache_candidate_scores = None  # 保存 LightGCN 得分
        self.action_list = None

        # 新增属性用于延迟反馈与缓存更新惩罚
        self.last_avg_delay = 0.0
        self.prev_cache = None

        self.current_step = 0
        self.max_steps = args.max_steps

    def update_candidate_set(self):
        """
        利用 LightGCN 预测用户对物品的评分，选取 topk_candidate 个物品作为候选缓存集合，
        并更新所有可能的缓存组合动作空间。
        """
        user_ids = [v.user_id for v in self.crossroad.vehicles]
        if len(user_ids) == 0:
            self.cache_candidate_set = np.arange(0, min(100, self.num_items))
            self.cache_candidate_scores = np.zeros(self.topk_candidate)
        else:
            user_tensor = torch.tensor(user_ids, dtype=torch.long).to(device=self.args.device)
            with torch.no_grad():
                user_embeds, item_embeds = self.recommender(self.norm_adj)
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
        self.prev_cache = self.current_cache

    def reset(self):
        self.crossroad.vehicles = []
        for i in range(self.args.num_vehicles):
            self.crossroad.generate_vehicle(user_id=i)
        self.update_candidate_set()
        self.current_step = 0
        self.last_avg_delay = 0.0
        obs = self._get_state()
        return obs

    def step(self, action):
        previous_cache = self.current_cache
        self.current_cache = self.action_list[action]
        cache_files = [int(self.cache_candidate_set[idx]) for idx in self.current_cache]

        total_reward = 0.0
        total_delay = 0.0
        hit_count = 0
        total_requests = 0

        # 遍历每辆车，根据它的请求频率发起请求
        for vehicle in self.crossroad.vehicles:
            num_requests = int(vehicle.request_frequency)
            for _ in range(num_requests):
                uid = vehicle.user_id
                if uid in self.test_user_ratings:
                    pos_items, pos_ratings = self.test_user_ratings[uid]
                else:
                    pos_items, pos_ratings = [], []

                # 如果启用推荐提升，则提高候选集合中物品的权重
                if self.use_recommendation_boost and self.cache_candidate_set is not None:
                    boost = self.args.recommendation_boost
                    # 对于 pos_items 中出现在候选集合的物品，提升其评分
                    pos_ratings = [r * boost if item in self.cache_candidate_set else r
                                   for item, r in zip(pos_items, pos_ratings)]

                requested_item = sample_request_item(pos_items, pos_ratings, self.zipf_s)
                if requested_item == -1:
                    continue

                distance = np.linalg.norm(vehicle.position - self.crossroad.rsu.position)
                normalized_distance = distance / self.crossroad.max_distance

                if requested_item in cache_files:
                    delay = self.args.base_delay - self.args.hit_delay_reduction * (1 - normalized_distance)
                    reward_request = self.args.hit_reward - self.args.delay_weight * delay
                    hit_count += 1
                    vehicle.request_frequency = min(vehicle.request_frequency + self.args.request_frequency_increment,
                                                    self.args.max_request_frequency)
                else:
                    delay = self.args.base_delay + self.args.miss_delay_penalty
                    reward_request = self.args.miss_penalty - self.args.delay_weight * delay
                    vehicle.request_frequency = max(vehicle.request_frequency - self.args.request_frequency_decay,
                                                    self.args.base_request_frequency)
                total_reward += reward_request
                total_delay += delay
                total_requests += 1

        if previous_cache != self.current_cache:
            total_reward -= self.args.cache_update_cost

        avg_delay = total_delay / total_requests if total_requests > 0 else 0.0
        avg_reward = total_reward / total_requests if total_requests > 0 else 0.0

        self.crossroad.simulate_step(dt=self.args.cross_dt)
        self.update_candidate_set()
        self.current_step += 1
        self.last_avg_delay = avg_delay
        next_state = self._get_state()
        done = (self.current_step >= self.max_steps)
        info = {
            "cache_hits": int(hit_count),
            "total_requests": int(total_requests),
            "hit_rate": float(hit_count / total_requests if total_requests > 0 else 0.0),
            "avg_delay": avg_delay
        }
        if self.current_step % 50 == 0:
            print(f"Step {self.current_step}, Avg Reward: {avg_reward:.4f}, Hit Rate: {info['hit_rate']:.4f}, "
                  f"Avg Delay: {avg_delay:.4f}, Cache Hits: {hit_count}/{total_requests}")
        return next_state, avg_reward, done, info

    def _get_state(self):
        if len(self.crossroad.vehicles) > 0:
            positions = np.array([v.position for v in self.crossroad.vehicles])
            speeds = np.array([v.speed for v in self.crossroad.vehicles])
            bandwidths = np.array([v.get_bandwidth(self.crossroad.rsu.position) for v in self.crossroad.vehicles])

            avg_position = positions.mean(axis=0) if positions.size > 0 else np.zeros(2)
            avg_speed = speeds.mean(axis=0) if speeds.size > 0 else np.zeros(2)
            avg_bandwidth = np.array([bandwidths.mean()]) if bandwidths.size > 0 else np.zeros(1)
        else:
            avg_position = np.zeros(2)
            avg_speed = np.zeros(2)
            avg_bandwidth = np.zeros(1)

        # Normalize position and speed values
        avg_position = avg_position / np.linalg.norm(avg_position) if np.linalg.norm(avg_position) > 0 else avg_position
        avg_speed = avg_speed / np.linalg.norm(avg_speed) if np.linalg.norm(avg_speed) > 0 else avg_speed

        # Normalize缓存状态（假设 num_items 很大）
        cache_state = np.array([self.cache_candidate_set[idx] / self.args.num_items for idx in self.current_cache])

        # Normalize candidate scores
        candidate_scores = np.array(self.cache_candidate_scores)
        if candidate_scores.max() > 1:
            candidate_scores = candidate_scores / candidate_scores.max()

        # 平均延迟作为状态的最后一项（可以进一步归一化）
        avg_delay_feature = np.array([self.last_avg_delay])

        # 确保所有数组均为 1D
        avg_position = avg_position.flatten()
        avg_speed = avg_speed.flatten()
        avg_bandwidth = avg_bandwidth.flatten()
        cache_state = cache_state.flatten()
        candidate_scores = candidate_scores.flatten()
        avg_delay_feature = avg_delay_feature.flatten()

        # 拼接并裁剪到 [-1, 1]
        state = np.concatenate([avg_position, avg_speed, avg_bandwidth, cache_state, candidate_scores, avg_delay_feature], axis=0)
        state = np.clip(state, -1.0, 1.0).astype(np.float32)

        return state
