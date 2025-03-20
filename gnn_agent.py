import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym.spaces import Dict, Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# 引入 torch_geometric 中的卷积模块
from torch_geometric.nn import GCNConv, GATConv, TransformerConv


# ------------------ TorchGeoGNN 模块 ------------------
class TorchGeoGNN(nn.Module):
    """
    使用 torch-geometric 实现的通用 GNN 模块，
    支持使用 'gcn'、'gat' 或 'transformer' 作为图卷积层。
    """

    def __init__(self, in_channels, hidden_channels, out_channels, conv_type="gcn"):
        super(TorchGeoGNN, self).__init__()
        self.conv_type = conv_type.lower()
        if self.conv_type == "gcn":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif self.conv_type == "gat":
            self.conv1 = GATConv(in_channels, hidden_channels, heads=1)
            self.conv2 = GATConv(hidden_channels, out_channels, heads=1)
        elif self.conv_type == "transformer":
            self.conv1 = TransformerConv(in_channels, hidden_channels)
            self.conv2 = TransformerConv(hidden_channels, out_channels)
        else:
            raise ValueError("不支持的 conv_type: {}".format(conv_type))

    def forward(self, x, edge_index):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # [num_nodes, out_channels]


# ------------------ TorchGeoGNN 特征提取器 ------------------
class TorchGeoGNNFeatureExtractor(BaseFeaturesExtractor):
    """
    使用 torch-geometric 构造的 GNN 特征提取器。
    要求环境观测为 Dict 格式，包含：
       - "node_features": Tensor, shape (B, max_nodes, node_feature_dim)
       - "edge_index": Tensor, shape (B, 2, max_edges)
       - "node_mask": Tensor, shape (B, max_nodes)
    输出全局特征向量，形状 (B, features_dim)，这里采用所有有效节点嵌入的平均池化。
    """
    def __init__(self, observation_space: Dict, features_dim: int = 64, conv_type="gcn"):
        super(TorchGeoGNNFeatureExtractor, self).__init__(observation_space, features_dim)
        self.num_nodes = observation_space.spaces["node_features"].shape[0]
        self.node_feature_dim = observation_space.spaces["node_features"].shape[1]
        self.gnn = TorchGeoGNN(self.node_feature_dim, hidden_channels=128, out_channels=features_dim,
                               conv_type=conv_type)

    def forward(self, observations):
        # 如果 batch_size 大于1，则逐个样本处理
        if observations["node_features"].dim() == 3:
            batch_size = observations["node_features"].shape[0]
            global_features = []
            for i in range(batch_size):
                # 单个样本的输入形状：
                # node_features: (max_nodes, node_feature_dim)
                # edge_index: (2, max_edges)
                # node_mask: (max_nodes,)
                x = observations["node_features"][i]  # (max_nodes, node_feature_dim)
                edge_index = observations["edge_index"][i]  # (2, max_edges)
                node_mask = observations["node_mask"][i]  # (max_nodes,)
                # 如果 edge_index 第一维不是2，则转置
                edge_index = edge_index.long()
                # 过滤掉 edge_index 中填充值 -1 的无效边
                valid_mask = (edge_index >= 0).all(dim=0)
                edge_index = edge_index[:, valid_mask]
                # 计算节点嵌入
                node_embeds = self.gnn(x, edge_index)  # (max_nodes, features_dim)
                # 将无效节点（mask==0）置零
                node_embeds = node_embeds * node_mask.unsqueeze(-1)
                valid_count = torch.sum(node_mask) + 1e-6
                global_feature = torch.sum(node_embeds, dim=0, keepdim=True) / valid_count  # (1, features_dim)
                global_features.append(global_feature)
            global_features = torch.cat(global_features, dim=0)  # (B, features_dim)
            return global_features
        else:
            # 处理单样本情况
            x = observations["node_features"].squeeze(0)  # (max_nodes, node_feature_dim)
            edge_index = observations["edge_index"].squeeze(0).long()  # (2, max_edges)
            node_mask = observations["node_mask"].squeeze(0)  # (max_nodes,)
            valid_mask = (edge_index >= 0).all(dim=0)
            edge_index = edge_index[:, valid_mask]
            node_embeds = self.gnn(x, edge_index)  # (max_nodes, features_dim)
            node_embeds = node_embeds * node_mask.unsqueeze(-1)
            valid_count = torch.sum(node_mask) + 1e-6
            global_feature = torch.sum(node_embeds, dim=0, keepdim=True) / valid_count  # (1, features_dim)
            return global_feature



# ------------------ 新的 PPO 策略 ------------------
class TorchGeoGNNPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, conv_type="gcn", **kwargs):
        super(TorchGeoGNNPPOPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=TorchGeoGNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64, conv_type=conv_type)
        )