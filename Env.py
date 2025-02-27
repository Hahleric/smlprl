import gym
from gym import spaces
import itertools
import numpy as np
import torch
from tqdm import tqdm

def sample_request_item(pos_items, pos_ratings, zipf_s=1.0):
    """
    根据用户在测试集中的正交互项目及对应评分，利用 Zipf 分布采样出一个请求项。
    :param pos_items: 用户交互的项目ID列表
    :param pos_ratings: 对应的评分列表
    :param zipf_s: Zipf 分布参数，s 越大越偏向评分最高的项目
    :return: 被采样的项目ID（int），如果用户无正交互返回 -1
    """
    if len(pos_items) == 0:
        return -1
    pos_items = np.array(pos_items)
    pos_ratings = np.array(pos_ratings, dtype=np.float32)
    sorted_idx = np.argsort(-pos_ratings)  # 按评分降序排序
    sorted_items = pos_items[sorted_idx]
    ranks = np.arange(1, len(sorted_items) + 1)  # 排名从1开始
    zipf_weights = 1.0 / (ranks ** zipf_s)
    probs = zipf_weights / zipf_weights.sum()
    chosen_idx = np.random.choice(len(sorted_items), p=probs)
    return int(sorted_items[chosen_idx])

class CarCachingEnv(gym.Env):
    """
    模拟路口中车辆、RSU 与缓存决策过程的环境。
    RL 代理的任务是选择缓存哪些物品，
    环境中每辆车根据自己在测试集上的交互（评分信息）通过 Zipf 分布采样出请求项，
    若请求命中当前 RSU 缓存则给予正奖励，否则给予惩罚。
    候选缓存集合由训练好的 LightGCN 模型根据当前车辆用户推荐结果（例如 Top20）生成。
    """
    def __init__(self, args, crossroad, recommender, norm_adj, num_items,
                 test_user_ratings, cache_capacity=5, zipf_s=0.8, topk_candidate=20):
        """
        :param crossroad: Crossroad 实例（包含车辆、RSU 及车辆运动逻辑）
        :param recommender: 训练好的 LightGCN 模型（GPU上）
        :param norm_adj: LightGCN 模型使用的归一化邻接矩阵（GPU上）
        :param num_items: 总物品数
        :param test_user_ratings: dict，映射用户ID -> (pos_items, pos_ratings)
        :param cache_capacity: RSU 的缓存容量（缓存物品数量）
        :param zipf_s: Zipf 分布参数，用于采样请求项
        :param topk_candidate: 用 LightGCN 生成候选缓存集合时取的 top-K 数
        """
        super(CarCachingEnv, self).__init__()
        self.args = args
        self.crossroad = crossroad
        self.recommender = recommender    # 已训练好的 LightGCN 模型
        self.norm_adj = norm_adj          # GPU 上的归一化邻接矩阵
        self.num_items = num_items
        self.test_user_ratings = test_user_ratings
        self.cache_capacity = cache_capacity
        self.zipf_s = zipf_s
        self.topk_candidate = topk_candidate

        # 初始时候候选集合和动作空间暂时为空，待 reset 时更新
        self.cache_candidate_set = None
        self.action_list = None

        # 状态空间由：车辆平均位置 (2)、平均速度 (2)、平均带宽 (1)、
        # 当前 RSU 缓存状态（cache_capacity 维，每个值为缓存物品ID归一化，范围[0,1]）
        self.feature_dim = 2 + 2 + 1 + self.cache_capacity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)  # 初始化时先设置为1

        # 当前 RSU 缓存配置（存储候选集合中的索引组合），初始为空
        self.current_cache = None

        # 奖励参数
        self.hit_reward = 1.0
        self.miss_penalty = -1.0
        self.num_requests_per_vehicle = 1
        self.max_steps = 150
        self.current_step = 0

    def update_candidate_set(self):
        """
        使用 LightGCN 模型根据当前环境中车辆的用户信息，
        计算所有物品得分（取所有车辆的用户预测得分平均），
        并选取 topk_candidate 个物品作为候选缓存集合。
        同时更新动作空间（所有从候选集合中选择 cache_capacity 个物品的组合）。
        """
        user_ids = [v.user_id for v in self.crossroad.vehicles]
        if len(user_ids) == 0:
            # 如果没有车辆，则使用默认候选集合
            self.cache_candidate_set = np.arange(0, min(100, self.num_items))
        else:
            user_tensor = torch.tensor(user_ids, dtype=torch.long).to(device=self.args.device)
            with torch.no_grad():
                user_embeds, item_embeds = self.recommender(self.norm_adj)
                # 对当前车辆用户的嵌入取平均，形成一个整体“兴趣向量”
                avg_user_embed = user_embeds[user_tensor].mean(dim=0, keepdim=True)  # (1, embedding_dim)
                scores = torch.matmul(avg_user_embed, item_embeds.t()).squeeze(0)  # (num_items,)
                # 选取 topk_candidate 个物品
                topk_items = torch.topk(scores, self.topk_candidate).indices.cpu().numpy()
                self.cache_candidate_set = topk_items
        # 更新动作空间：候选集合中选择 cache_capacity 个物品的所有组合
        self.action_list = list(itertools.combinations(range(len(self.cache_candidate_set)), self.cache_capacity))
        self.num_actions = len(self.action_list)
        self.action_space = spaces.Discrete(self.num_actions)
        # 默认初始缓存选择第一个组合
        self.current_cache = self.action_list[0]

    def reset(self):
        # 重置环境：清空车辆并生成新车辆（例如20辆，每辆车对应一个 user_id）
        self.crossroad.vehicles = []
        for i in range(20):
            self.crossroad.generate_vehicle(user_id=i)
        # 更新候选集合与动作空间
        self.update_candidate_set()
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        # 更新缓存配置
        self.update_candidate_set()
        self.current_cache = self.action_list[action]
        cache_files = [int(self.cache_candidate_set[idx]) for idx in self.current_cache]

        total_reward = 0.0
        hit_count = 0
        delay_penalty = 0.0  # 额外的基于延迟的惩罚

        # 预计算 LightGCN 产生的用户嵌入
        with torch.no_grad():
            user_embeds, item_embeds = self.recommender(self.norm_adj)

        # 遍历所有车辆
        for vehicle in tqdm(self.crossroad.vehicles, desc="Processing vehicles", leave=False):
            uid = vehicle.user_id
            user_emb = user_embeds[uid]

            for _ in range(self.num_requests_per_vehicle):
                if uid in self.test_user_ratings:
                    pos_items, pos_ratings = self.test_user_ratings[uid]
                else:
                    pos_items, pos_ratings = [], []

                requested_item = sample_request_item(pos_items, pos_ratings, self.zipf_s)
                if requested_item == -1:
                    continue

                # 计算车辆到 RSU 的带宽和距离
                bandwidth = vehicle.get_bandwidth(self.crossroad.rsu.position)
                distance = np.linalg.norm(vehicle.position - self.crossroad.rsu.position)
                normalized_distance = distance / self.crossroad.max_distance  # 归一化距离
                delay = 1 / (bandwidth + 1e-6)  # 避免除零

                if requested_item in cache_files:
                    # 命中缓存，根据 QoE 计算奖励
                    qoe_gain = 1.0 - normalized_distance  # 近距离时 QoE 提升更大
                    total_reward += 1 + qoe_gain  # 额外奖励 QoE 提升
                    hit_count += 1
                else:
                    # 未命中，计算动态惩罚
                    delay_penalty = min(1.0, delay)  # 限制最大惩罚
                    total_reward -= (0.1 + delay_penalty)  # 综合惩罚

        num_total = self.num_requests_per_vehicle * len(self.crossroad.vehicles)
        avg_reward = total_reward / num_total if num_total > 0 else 0.0

        # 模拟下一步
        self.crossroad.simulate_step(dt=1.0)
        self.current_step += 1
        next_state = self._get_state()
        done = (self.current_step >= self.max_steps)

        # 记录统计信息
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
        state = np.concatenate([avg_position, avg_speed, avg_bandwidth, cache_state])
        return state.astype(np.float32)
