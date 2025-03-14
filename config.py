import argparse

def get_config():
    parser = argparse.ArgumentParser(description="车联网缓存问题中使用 RL 与 LightGCN 的配置")

    # 数据及预处理相关
    parser.add_argument('--data', type=str, default='ml-1m', help='使用的数据集，默认 ml-1m')
    parser.add_argument('--ratings_path', type=str, default='ml-1m/ratings.dat', help='评分数据文件路径')
    parser.add_argument('--delimiter', type=str, default='::', help='评分数据的分隔符')
    parser.add_argument('--min_rating_threshold', type=float, default=0.0, help='评分过滤阈值')
    parser.add_argument('--zipf_s', type=float, default=1.0, help='Zipf 分布参数（默认备用，实际使用 rl_zipf_s 或 trad_zipf_s）')
    parser.add_argument('--rl_zipf_s', type=float, default=1.0, help='RL 环境中 Zipf 分布参数')
    parser.add_argument('--trad_zipf_s', type=float, default=1.0, help='传统策略中 Zipf 分布参数')
    parser.add_argument('--pca_components', type=int, default=10, help='PCA 降维后的维度，用于生成用户兴趣向量')
    parser.add_argument('--episodes', type=int, default=10, help='训练的总 episode 数')
    parser.add_argument('--use_sbl', type=bool, default=True, help='是否使用 SBL 进行预训练')
    parser.add_argument('--num_items', type=int, default=3952, help='物品总数')
    parser.add_argument('--recommendation_boost', type=float, default=1.5, help='推荐物品的额外权重')
    parser.add_argument('--recommendation_boost_trad', type=bool, default=False, help='是否在传统策略中使用推荐物品权重')
    parser.add_argument('--use_saved_rl', type=bool, default=True, help='是否使用保存的 RL 模型')

    # RL 环境相关
    parser.add_argument('--num_vehicles', type=int, default=10, help='路口中的车辆数')
    parser.add_argument('--cache_capacity', type=int, default=8, help='RSU 缓存容量，即可缓存的物品数量')
    parser.add_argument('--num_requests_per_vehicle', type=int, default=1, help='默认每个车辆每个时间步发起的请求数（备用）')
    parser.add_argument('--max_steps', type=int, default=2048, help='每个 episode 的最大步数')
    parser.add_argument('--testing_step', type=int, default=50, help='测试时的步数')
    parser.add_argument('--hit_threshold', type=float, default=100, help='命中率阈值, 用于奖励')
    parser.add_argument('--cross_dt', type=float, default=5, help='路口模拟的时间步长')
    parser.add_argument('--spawn_rate', type=float, default=1.2, help='每个时间步生成新车辆的期望数量')

    # 新增网络延迟与奖励相关参数
    parser.add_argument('--base_delay', type=float, default=1.0, help='基础延迟（单位可自定义）')
    parser.add_argument('--hit_delay_reduction', type=float, default=0.3, help='缓存命中时减少的延迟')
    parser.add_argument('--miss_delay_penalty', type=float, default=0.7, help='缓存未命中时增加的延迟')
    parser.add_argument('--delay_weight', type=float, default=10.0, help='延迟对奖励的权重')
    parser.add_argument('--hit_reward', type=float, default=5.0, help='缓存命中时的奖励基值')
    parser.add_argument('--miss_penalty', type=float, default=-20.0, help='缓存未命中时的惩罚基值')
    parser.add_argument('--cache_update_cost', type=float, default=2, help='每次更新缓存的惩罚成本')

    # 用户请求行为相关
    parser.add_argument('--base_request_frequency', type=float, default=1.0, help='车辆初始请求频率')
    parser.add_argument('--request_frequency_increment', type=float, default=0.5, help='缓存命中时增加的请求频率')
    parser.add_argument('--request_frequency_decay', type=float, default=0.1, help='缓存未命中时衰减的请求频率')
    parser.add_argument('--max_request_frequency', type=float, default=5.0, help='请求频率的最大值')
    parser.add_argument('--min_request_frequency', type=float, default=0.5, help='请求频率的最小值')

    # 推荐模型 LightGCN 相关
    parser.add_argument('--embedding_dim', type=int, default=64, help='LightGCN 用户和物品嵌入的维度')
    parser.add_argument('--num_layers', type=int, default=3, help='LightGCN 传播层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='LightGCN 的 dropout 概率')

    # RL Agent 超参数
    parser.add_argument('--use_gnn', type=bool, default=False, help='是否使用 GNN 作为特征提取器')
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
    parser.add_argument('--sbl_device', type=str, default='cpu', help='SBL 预训练时使用的设备')

    config = parser.parse_args()
    return config
