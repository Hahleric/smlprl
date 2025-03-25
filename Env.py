from typing import Optional, Union, List
import gym
from gym import spaces
import itertools
import numpy as np
import torch
from gym.core import RenderFrame
import math

from gym.spaces import Dict, Box, MultiDiscrete
from matplotlib import pyplot as plt


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
        self.last_info = {}
        self.fig, self.ax = None, None
        self.args = args
        self.crossroad = crossroad
        self.recommender = recommender.eval()  # 训练好的 LightGCN 模型
        self.norm_adj = norm_adj  # 归一化邻接矩阵（应位于 args.device 上）
        self.num_items = num_items
        self.test_user_ratings = test_user_ratings
        self.cache_capacity = cache_capacity
        self.zipf_s = zipf_s
        self.topk_candidate = topk_candidate
        self.use_recommendation_boost = use_recommendation_boost

        # 状态空间：车辆平均位置(2) + 平均速度(2) + 平均带宽(1) + 当前缓存状态(cache_capacity)
        # + 当前候选集合的得分(topk_candidate) + 平均延迟(1)
        # self.feature_dim = 2 + 2 + 1 + self.cache_capacity + self.topk_candidate + 1
        self.feature_dim = self.topk_candidate + 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.feature_dim,), dtype=np.float32)

        # 修改动作空间：
        # 原来的动作空间是 MultiDiscrete([topk_candidate] * cache_capacity)，代表直接选择 cache_capacity 个候选项索引。
        # 这里修改为 Box 空间，形状为 (topk_candidate,)，代表对整个候选集合的打分，
        # 最终通过排序选择出得分最高的 cache_capacity 个候选项作为缓存配置。
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.topk_candidate,), dtype=np.float32)

        self.current_cache = None  # 当前缓存配置，保存候选集合中的索引列表
        self.cache_candidate_set = None
        self.cache_candidate_scores = None  # 保存 LightGCN 得分

        # 新增属性用于延迟反馈与缓存更新惩罚
        self.last_avg_delay = 0.0
        self.prev_cache = None

        self.current_step = 0
        self.max_steps = args.max_steps

    def update_candidate_set(self):
        """
        利用 LightGCN 预测用户对物品的评分，选取 topk_candidate 个物品作为候选缓存集合，
        并更新候选集合（不再枚举所有组合）。
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

        # 默认初始缓存选择：直接选择候选集合前 cache_capacity 个文件
        if self.current_cache is None:
            self.current_cache = np.arange(self.cache_capacity)

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
        # 这里 action 现在为一个长度为 topk_candidate 的连续向量，
        # 每个值代表对候选集合中对应物品的打分。
        action = np.array(action).flatten()
        if action.shape[0] != self.topk_candidate:
            raise ValueError(f"Action must be of shape ({self.topk_candidate},) but got {action.shape}")
        # 对候选集合打分进行降序排序，选取得分最高的 cache_capacity 个索引
        ranking_indices = np.argsort(-action)
        selected_indices = ranking_indices[:self.cache_capacity].tolist()
        self.current_cache = selected_indices

        # 根据当前候选集合索引得到真实物品 id 列表
        cache_files = [int(self.cache_candidate_set[idx]) for idx in self.current_cache]

        total_reward = 0.0
        total_delay = 0.0
        hit_count = 0
        total_requests = 0
        k = 2
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

                    reward_scale = 1 / (1 + math.exp(-k * (self.args.base_delay - delay)))
                    hit_count += 1
                    vehicle.request_frequency = min(vehicle.request_frequency + self.args.request_frequency_increment,
                                                    self.args.max_request_frequency)
                    reward_request = self.args.hit_reward * reward_scale - self.args.delay_weight * delay
                else:
                    delay = self.args.base_delay + self.args.miss_delay_penalty
                    reward_scale = 1 / (1 + math.exp(-k * (self.args.base_delay - delay)))
                    vehicle.request_frequency = max(vehicle.request_frequency - self.args.request_frequency_decay,
                                                    self.args.base_request_frequency)
                    reward_request = self.args.miss_penalty * (1 - reward_scale) - self.args.delay_weight * delay
                total_reward += reward_request
                total_delay += delay
                total_requests += 1
        total_reward = total_reward / total_requests
        if not np.array_equal(previous_cache, self.current_cache):
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
            "avg_delay": avg_delay,
            "avg_reward": avg_reward
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
        # avg_speed = avg_speed / np.linalg.norm(avg_speed) if np.linalg.norm(avg_speed) > 0 else avg_speed

        # 当前缓存状态：将当前缓存的候选项（item id）归一化
        cache_state = np.array([self.cache_candidate_set[idx] / self.args.num_items for idx in self.current_cache])

        # 使用归一化后的候选集合 item id 作为状态的一部分
        # self.cache_candidate_set 的长度为 topk_candidate
        normalized_candidate_ids = np.array(self.cache_candidate_set, dtype=np.float32) / self.args.num_items

        # 平均延迟作为状态的最后一项（可以进一步归一化）
        avg_delay_feature = np.array([self.last_avg_delay])

        # 确保所有数组均为 1D
        # avg_position = avg_position.flatten()
        # avg_speed = avg_speed.flatten()
        # avg_bandwidth = avg_bandwidth.flatten()
        normalized_candidate_scores = self.cache_candidate_scores.flatten()
        # cache_state = cache_state.flatten()
        normalized_candidate_ids = normalized_candidate_ids.flatten()
        avg_delay_feature = avg_delay_feature.flatten()

        # 拼接状态，这里我们用候选 item id 替换了之前的候选得分
        # 注意：normalized_candidate_ids 部分使得模型可以知道候选集合的情况，从而学习对其排序
        state = np.concatenate(
            # [avg_position, avg_speed, avg_bandwidth, cache_state, normalized_candidate_ids, avg_delay_feature], axis=0)
            [normalized_candidate_scores, avg_delay_feature], axis=0)
        state = np.clip(state, -1.0, 1.0).astype(np.float32)

        return state


def normalize_node_features(node_features, env):
    """
    对节点特征进行归一化：
      - 对位置（前两列）和速度（第3、4列）分别归一化；
      - 带宽直接 clip 到 [-1,1]；
      - 对 request_frequency（第6列），使用 env.args 中 base_request_frequency 与 max_request_frequency 归一化到 [-1,1]；
      - 第7列保持不变。
    """
    normed = np.copy(node_features)
    for i in range(normed.shape[0]):
        # 归一化位置
        pos = normed[i, 0:2]
        norm = np.linalg.norm(pos)
        if norm > 0:
            normed[i, 0:2] = pos / norm
        # 归一化速度
        sp = normed[i, 2:4]
        norm_sp = np.linalg.norm(sp)
        if norm_sp > 0:
            normed[i, 2:4] = sp / norm_sp
        # 带宽 clip
        normed[i, 4] = np.clip(normed[i, 4], -1.0, 1.0)
        # 归一化 request_frequency（第6列）
        base_rf = env.args.base_request_frequency
        max_rf = env.args.max_request_frequency if hasattr(env.args, "max_request_frequency") else 10.0
        normed[i, 5] = 2 * (normed[i, 5] - base_rf) / (max_rf - base_rf) - 1
        # 常数列保持不变
    normed = np.clip(normed, -1.0, 1.0)
    return normed

class GNNCarCachingEnv(CarCachingEnv):
    def __init__(self, *args, **kwargs):
        super(GNNCarCachingEnv, self).__init__(*args, **kwargs)
        # 固定最大节点数：RSU + 最大车辆数（由配置参数 gnn_max_vehicles 指定）
        max_nodes = self.args.gnn_max_vehicles + 1
        node_feature_dim = 7
        # 固定最大边数：对于星型拓扑，最大边数 = 自连接 max_nodes + 2*(max_nodes - 1) = 3*max_nodes - 2
        max_edges = 3 * max_nodes - 2
        from gym.spaces import Dict, Box
        self.observation_space = Dict({
            "node_features": Box(low=-1.0, high=1.0, shape=(max_nodes, node_feature_dim), dtype=np.float32),
            "edge_index": Box(low=-1, high=max_nodes, shape=(2, max_edges), dtype=np.int64),
            "node_mask": Box(low=0.0, high=1.0, shape=(max_nodes,), dtype=np.float32)
        })
        out_dim = 16  # 输出的节点嵌入维度
        self.gnn_state_dim = out_dim

    def _get_state_gnn(self):
        """
        构建图：节点包括 RSU 与所有车辆
          - 节点特征：车辆节点 [x, y, vx, vy, 带宽, request_frequency, 1]；RSU节点 [x, y, 0, 0, avg_bandwidth, 0, 1]
          - 使用 normalize_node_features 对节点特征归一化
          - 对节点特征进行 padding，使得形状固定为 (max_nodes, 7)，同时生成 node_mask
          - 边连接：基于实际节点构造边（自连接 + RSU 与车辆双向连边），实际边数 = 3*n_actual - 2，
            然后对 edge_index 进行 padding 至 (2, max_edges)，填充值 -1 表示无效边
        返回字典：{"node_features": Tensor, "edge_index": Tensor, "node_mask": Tensor}
        """
        device = self.args.device
        vehicles = self.crossroad.vehicles
        n_actual = len(vehicles) + 1  # 实际节点数：RSU + 实际车辆数
        max_nodes = self.args.gnn_max_vehicles + 1

        # RSU 节点特征
        rsu_pos = self.crossroad.rsu.position
        if vehicles:
            avg_bw = np.mean([v.get_bandwidth(self.crossroad.rsu.position) for v in vehicles])
        else:
            avg_bw = 0.0
        rsu_feature = np.array([rsu_pos[0], rsu_pos[1], 0.0, 0.0, avg_bw, 0.0, 1.0], dtype=np.float32)
        # 车辆节点特征
        vehicle_features = []
        for v in vehicles:
            bw = v.get_bandwidth(self.crossroad.rsu.position)
            feat = np.concatenate([v.position, v.speed, [bw], [v.request_frequency], [1.0]]).astype(np.float32)
            vehicle_features.append(feat)
        if vehicle_features:
            vehicle_features = np.stack(vehicle_features, axis=0)
        else:
            vehicle_features = np.empty((0, 7), dtype=np.float32)
        # 拼接 RSU 与车辆节点 -> shape: (n_actual, 7)
        node_features = np.concatenate([rsu_feature.reshape(1, -1), vehicle_features], axis=0)
        node_features = normalize_node_features(node_features, self)
        # Padding node_features 到 (max_nodes, 7)
        if n_actual < max_nodes:
            pad = np.zeros((max_nodes - n_actual, node_features.shape[1]), dtype=np.float32)
            node_features = np.concatenate([node_features, pad], axis=0)
        else:
            node_features = node_features[:max_nodes, :]
        node_features = torch.FloatTensor(node_features).to(device)

        # 生成 node_mask: 前 n_actual 个为1，其余为0，形状 (max_nodes,)
        node_mask = np.zeros(max_nodes, dtype=np.float32)
        node_mask[:n_actual] = 1.0
        node_mask = torch.FloatTensor(node_mask).to(device)

        # 构造 edge_index：先构造实际节点的边
        edge_index_list = []
        # 自连接：每个实际节点连边
        for i in range(n_actual):
            edge_index_list.append([i, i])
        # RSU 与每个车辆双向连边（RSU索引=0）
        for i in range(1, n_actual):
            edge_index_list.append([0, i])
            edge_index_list.append([i, 0])

        edge_index_arr = np.array(edge_index_list).T  # shape: (2, actual_edges)
        actual_edges = edge_index_arr.shape[1]
        max_edges = 3 * max_nodes - 2
        if actual_edges < max_edges:
            pad = -1 * np.ones((2, max_edges - actual_edges), dtype=np.int64)
            edge_index_arr = np.concatenate([edge_index_arr, pad], axis=1)
        else:
            edge_index_arr = edge_index_arr[:, :max_edges]
        edge_index = torch.LongTensor(edge_index_arr).to(device)
        return {"node_features": node_features, "edge_index": edge_index, "node_mask": node_mask}

    def _get_state(self):
        if self.args.use_gnn:
            return self._get_state_gnn()
        else:
            return super(GNNCarCachingEnv, self)._get_state()
