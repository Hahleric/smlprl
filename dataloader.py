import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader


class MovieLensDataLoader:
    """
    该类用于加载 ml-1m 数据集，并进行预处理：
      1. 计算电影流行度（先使用时间衰减统计评分权重，再利用 Zipf 分布调整）
      2. 构造用户评分矩阵，并使用 PCA 降维，生成用户兴趣向量
    """

    def __init__(self,
                 ratings_path='ml-1m/ratings.dat',
                 delimiter='::',
                 zipf_s=1.0,
                 pca_components=10,
                 min_rating_threshold=0.0):
        """
        :param ratings_path: 评分数据文件路径（ml-1m 文件格式：UserID::MovieID::Rating::Timestamp）
        :param delimiter: 数据分隔符（ml-1m 使用 "::"）
        :param zipf_s: Zipf 分布参数 s，控制权重衰减
        :param pca_components: PCA 降维后维度
        :param min_rating_threshold: 若需要过滤评分低于某值的数据（默认不过滤）
        """
        self.ratings_path = ratings_path
        self.delimiter = delimiter
        self.zipf_s = zipf_s
        self.pca_components = pca_components
        self.min_rating_threshold = min_rating_threshold

        # 加载原始数据
        self.ratings = self._load_ratings()
        # 获取当前时间戳（这里取数据中的最大时间戳作为参考）
        self.current_time = self.ratings['Timestamp'].max()
        # 预处理：计算电影流行度 & 用户兴趣向量
        self.movie_popularity = self._compute_movie_popularity()
        self.user_interest_vectors = self._compute_user_interest_vectors()

    def _load_ratings(self):
        """加载 ml-1m 评分数据"""
        if not os.path.exists(self.ratings_path):
            raise FileNotFoundError(f"找不到评分文件：{self.ratings_path}")

        ratings = pd.read_csv(self.ratings_path,
                              sep=self.delimiter,
                              engine='python',
                              names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        # 可选：过滤低评分数据
        if self.min_rating_threshold > 0:
            ratings = ratings[ratings['Rating'] >= self.min_rating_threshold]
        return ratings

    def _compute_movie_popularity(self):
        """
        计算电影流行度：
          1. 对于每条评分，根据时间戳衰减计算权重（较新的评分权重更高）
          2. 对每个电影求和得到原始热度
          3. 根据原始热度降序排列，再利用 Zipf 分布公式调整（权重 = 1/(rank^s)）
          4. 最后归一化得到概率分布
        """
        lambda_decay = 1e-7  # 衰减参数，根据数据范围调整
        # 计算每条评分的衰减权重
        self.ratings['DecayWeight'] = np.exp(-lambda_decay * (self.current_time - self.ratings['Timestamp']))

        # 统计每个电影的总权重
        movie_weight = self.ratings.groupby('MovieID')['DecayWeight'].sum()
        # 按权重降序排列
        movie_weight_sorted = movie_weight.sort_values(ascending=False)
        # 利用 Zipf 分布计算权重
        zipf_weights = {}
        for rank, movie_id in enumerate(movie_weight_sorted.index, start=1):
            zipf_weights[movie_id] = 1.0 / (rank ** self.zipf_s)
        # 转换为 Series并归一化
        zipf_series = pd.Series(zipf_weights)
        zipf_series = zipf_series / zipf_series.sum()
        return zipf_series

    def _compute_user_interest_vectors(self):
        """
        构造用户-电影评分矩阵，然后使用 PCA 降维
          - 构造矩阵：行为用户，列为电影，数值为评分（没有评分填 0）
          - 采用 sklearn.decomposition.PCA 将高维评分向量降至 self.pca_components 维
          - 返回字典：{user_id: interest_vector (ndarray)}
        """
        # 先获取所有用户和电影的列表
        all_users = self.ratings['UserID'].unique()
        all_movies = self.ratings['MovieID'].unique()
        all_movies.sort()  # 排序，保证维度顺序一致

        # 构造评分矩阵
        user_movie_matrix = pd.pivot_table(self.ratings,
                                           index='UserID',
                                           columns='MovieID',
                                           values='Rating',
                                           fill_value=0)
        # 保证所有电影都存在（缺失值填0）
        user_movie_matrix = user_movie_matrix.reindex(columns=all_movies, fill_value=0)
        rating_matrix = user_movie_matrix.values  # 形状为 (num_users, num_movies)

        # 使用 PCA 降维
        pca = PCA(n_components=self.pca_components)
        reduced_matrix = pca.fit_transform(rating_matrix)
        # 构造用户兴趣向量字典
        user_ids = user_movie_matrix.index.values
        user_interest = {user_id: reduced_matrix[i, :] for i, user_id in enumerate(user_ids)}
        return user_interest

    def get_user_dataset(self):
        """
        返回一个自定义的 PyTorch 数据集，用于封装用户兴趣向量
        """
        return MovieLensDataset(self.user_interest_vectors)

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=0):
        """
        返回一个 PyTorch DataLoader 对象，用于加载用户兴趣向量数据
        """
        dataset = self.get_user_dataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class MovieLensDataset(Dataset):
    """
    自定义数据集，每个样本包含 (user_id, interest_vector)
    interest_vector 为 PCA 降维后的用户兴趣向量
    """

    def __init__(self, user_interest_dict):
        """
        :param user_interest_dict: 字典，键为 user_id，值为兴趣向量（ndarray）
        """
        self.user_ids = list(user_interest_dict.keys())
        self.interest_vectors = [user_interest_dict[uid] for uid in self.user_ids]

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        # 转换为 torch.Tensor
        interest_vector = torch.tensor(self.interest_vectors[index], dtype=torch.float32)
        return user_id, interest_vector


# --------------------------
# 示例使用
# --------------------------
if __name__ == '__main__':
    # 初始化数据加载器（请确认 ratings.dat 文件路径正确）
    data_loader = MovieLensDataLoader(ratings_path='ml-1m/ratings.dat',
                                      delimiter='::',
                                      zipf_s=1.0,
                                      pca_components=10)

    # 查看预处理结果：电影流行度（Zipf 调整后）
    print("电影流行度（Zipf 调整后，归一化）：")
    print(data_loader.movie_popularity.head(10))

    # 获取用户兴趣数据集和 DataLoader
    dataset = data_loader.get_user_dataset()
    print(f"用户数量：{len(dataset)}")

    loader = data_loader.get_dataloader(batch_size=16, shuffle=True)
    for batch_idx, (user_ids, interest_vectors) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print("User IDs:", user_ids)
        print("Interest vectors shape:", interest_vectors.shape)
        # 这里只显示一个 batch
        break
