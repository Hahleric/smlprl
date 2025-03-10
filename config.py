# config.py
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="车联网缓存问题中使用 RL 与 LightGCN 的配置")

    # 数据及预处理相关
    parser.add_argument('--data', type=str, default='ml-1m', help='使用的数据集，默认 ml-1m')
    parser.add_argument('--ratings_path', type=str, default='ml-1m/ratings.dat', help='评分数据文件路径')
    parser.add_argument('--delimiter', type=str, default='::', help='评分数据的分隔符')
    parser.add_argument('--min_rating_threshold', type=float, default=0.0, help='评分过滤阈值')
    parser.add_argument('--zipf_s', type=float, default=0.8, help='Zipf 分布参数，控制评分权重的衰减')
    parser.add_argument('--pca_components', type=int, default=10, help='PCA 降维后的维度，用于生成用户兴趣向量')
    parser.add_argument('--episodes', type=int, default=10, help='训练的总 episode 数')

    # RL 环境相关
    parser.add_argument('--num_vehicles', type=int, default=20, help='路口中的车辆数')
    parser.add_argument('--cache_capacity', type=int, default=10, help='RSU 缓存容量，即可缓存的物品数量')
    parser.add_argument('--num_requests_per_vehicle', type=int, default=1, help='每个车辆每个时间步发起的请求数')
    parser.add_argument('--max_steps', type=int, default=50, help='每个 episode 的最大步数')
    parser.add_argument('--hit_threshold', type=float, default=100, help='命中率阈值, 用于奖励')
    parser.add_argument('--cross_dt', type=float, default=5, help='路口模拟的时间步长')
    parser.add_argument('--spawn_rate', type=float, default=1.5, help='每个时间步生成新车辆的期望数量')

    # 推荐模型 LightGCN 相关
    parser.add_argument('--embedding_dim', type=int, default=64, help='LightGCN 用户和物品嵌入的维度')
    parser.add_argument('--num_layers', type=int, default=3, help='LightGCN 传播层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='LightGCN 的 dropout 概率')

    # RL Agent（例如 A2C 或 DQN）超参数
    parser.add_argument('--lr', type=float, default=1e-4, help='RL 模型的基础学习率')
    parser.add_argument('--gamma', type=float, default=0.01, help='折扣因子')
    parser.add_argument('--epsilon', type=float, default=1.0, help='初始 epsilon 值（DQN 使用）')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='epsilon 衰减率')
    parser.add_argument('--epsilon_min', type=float, default=0.05, help='epsilon 最低值')
    parser.add_argument('--buffer_size', type=int, default=1000, help='经验回放缓冲区大小')
    parser.add_argument('--batch_size', type=int, default=64, help='训练时每批样本大小')

    # 如果使用 A2C 的话，可以单独设置 actor 和 critic 的学习率以及熵正则项
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='Actor 网络的学习率')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='Critic 网络的学习率')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='熵正则化系数，鼓励探索')
    parser.add_argument('--agent', type=str, default='ppo', help='RL agent going to be used')

    # 设备设置
    parser.add_argument('--device', type=str, default='mps', help='使用的设备，默认为 mps，如果不可用则为 cpu')

    config = parser.parse_args()
    return config

