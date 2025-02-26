import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.sparse as sp


#####################################
# 1. 数据加载与预处理 (ML-1M)
#####################################
def load_ml1m(filepath):
    # ml-1m 数据格式：UserID::MovieID::Rating::Timestamp
    df = pd.read_csv(filepath, sep="::", engine="python", header=None,
                     names=["UserID", "MovieID", "Rating", "Timestamp"])
    # 将ID转换为0-indexed
    df['UserID'] = df['UserID'] - 1
    df['MovieID'] = df['MovieID'] - 1
    return df


def build_user_item_matrix(df, num_users, num_items, split_ratio=0.8, seed=42):
    """
    对每个用户随机划分训练和测试评分，将评分>0视为正交互，构造二值矩阵
    """
    train_matrix = np.zeros((num_users, num_items), dtype=np.float32)
    test_matrix = np.zeros((num_users, num_items), dtype=np.float32)

    rng = np.random.default_rng(seed)
    for user, group in df.groupby("UserID"):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        split_point = int(len(indices) * split_ratio)
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        for idx in train_indices:
            u = int(df.loc[idx, "UserID"])
            i = int(df.loc[idx, "MovieID"])
            train_matrix[u, i] = 1.0
        for idx in test_indices:
            u = int(df.loc[idx, "UserID"])
            i = int(df.loc[idx, "MovieID"])
            test_matrix[u, i] = 1.0
    return train_matrix, test_matrix


def create_norm_adj_matrix(train_matrix):
    """
    构造 LightGCN 所需的归一化邻接矩阵（用户+项目二部图）
    """
    num_users, num_items = train_matrix.shape
    R = sp.csr_matrix(train_matrix)
    # 构造二部图：矩阵 A = [0, R; R^T, 0]
    upper = sp.hstack([sp.csr_matrix((num_users, num_users)), R])
    lower = sp.hstack([R.transpose(), sp.csr_matrix((num_items, num_items))])
    A = sp.vstack([upper, lower])
    A = A.tocoo()

    # 归一化处理 D^(-1/2)*A*D^(-1/2)
    rowsum = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocoo()

    # 转换为 torch.sparse.FloatTensor 并移动到GPU
    indices = torch.from_numpy(np.vstack((A_norm.row, A_norm.col)).astype(np.int64))
    values = torch.from_numpy(A_norm.data.astype(np.float32))
    shape = torch.Size(A_norm.shape)
    norm_adj = torch.sparse.FloatTensor(indices, values, shape).cuda()
    return norm_adj


#####################################
# 2. 定义 LightGCN 模型
#####################################
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, dropout=0.0):
        """
        :param num_users: 用户数量
        :param num_items: 项目数量
        :param embedding_dim: 嵌入维度
        :param num_layers: 传播层数
        :param dropout: dropout 概率
        """
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # 初始化用户和项目嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, norm_adj):
        # 拼接用户和项目初始嵌入 (shape: (num_users+num_items, embedding_dim))
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]

        for _ in range(self.num_layers):
            # 图传播：使用稀疏矩阵乘法
            all_embeddings = torch.sparse.mm(norm_adj, all_embeddings)
            if self.dropout:
                all_embeddings = F.dropout(all_embeddings, self.dropout, training=self.training)
            embeddings_list.append(all_embeddings)
        # 平均所有层的嵌入
        final_embedding = sum(embeddings_list) / (self.num_layers + 1)
        # 分割用户和项目嵌入
        user_final, item_final = torch.split(final_embedding, [self.num_users, self.num_items], dim=0)
        return user_final, item_final

    def get_score(self, user_indices, item_indices, norm_adj):
        user_final, item_final = self.forward(norm_adj)
        u_emb = user_final[user_indices]
        i_emb = item_final[item_indices]
        scores = (u_emb * i_emb).sum(dim=1)
        return scores


#####################################
# 3. 定义训练数据集 (BPR 负采样)
#####################################
class BPRDataset(torch.utils.data.Dataset):
    def __init__(self, train_matrix):
        """
        :param train_matrix: numpy 数组，形状 (num_users, num_items)，二值化评分矩阵
        """
        self.train_matrix = train_matrix
        self.num_users, self.num_items = train_matrix.shape
        self.user_item_pairs = []
        for u in range(self.num_users):
            pos_items = np.where(train_matrix[u] > 0)[0]
            for i in pos_items:
                self.user_item_pairs.append((u, i))

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        u, pos = self.user_item_pairs[idx]
        while True:
            neg = np.random.randint(0, self.num_items)
            if self.train_matrix[u, neg] == 0:
                break
        return u, pos, neg


def bpr_loss(u_emb, pos_emb, neg_emb):
    diff = (u_emb * (pos_emb - neg_emb)).sum(dim=1)
    loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
    return loss


#####################################
# 4. 训练与评估函数
#####################################
def train_lightgcn(model, norm_adj, train_dataset, num_epochs=10, batch_size=1024, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_loader:
            u, pos, neg = batch
            u = u.cuda()
            pos = pos.cuda()
            neg = neg.cuda()

            optimizer.zero_grad()
            user_embeds, item_embeds = model(norm_adj)
            u_emb = user_embeds[u]
            pos_emb = item_embeds[pos]
            neg_emb = item_embeds[neg]
            loss = bpr_loss(u_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * u.size(0)
        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")


def recall_ndcg_at_k(pred, ground_truth, k):
    pred = pred[:k]
    hit_count = len(set(pred) & set(ground_truth))
    recall = hit_count / float(len(ground_truth))
    dcg = 0.0
    for i, item in enumerate(pred):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg


def evaluate_lightgcn(model, norm_adj, train_matrix, test_matrix, topk=20):
    model.eval()
    with torch.no_grad():
        user_embeds, item_embeds = model(norm_adj)
        all_scores = torch.matmul(user_embeds, item_embeds.t())  # shape: (num_users, num_items)
        # 屏蔽训练集中的交互
        train_tensor = torch.from_numpy(train_matrix).cuda()
        all_scores[train_tensor > 0] = -1e8
        all_scores_np = all_scores.cpu().numpy()
        test_np = test_matrix
    num_users = train_matrix.shape[0]
    recall_list = []
    ndcg_list = []
    for u in range(num_users):
        gt_items = np.where(test_np[u] > 0)[0]
        if len(gt_items) == 0:
            continue
        topk_indices = np.argsort(-all_scores_np[u])[:topk]
        rec, ndcg = recall_ndcg_at_k(topk_indices, gt_items, topk)
        recall_list.append(rec)
        ndcg_list.append(ndcg)
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
    print(f"Recall@{topk}: {avg_recall:.4f}, NDCG@{topk}: {avg_ndcg:.4f}")
    return avg_recall, avg_ndcg


#####################################
# 5. 主流程（全部运行在GPU上）
#####################################
if __name__ == '__main__':
    # 检查GPU是否可用
    if not torch.cuda.is_available():
        print("GPU不可用，请检查CUDA环境。")
        exit(1)

    # 加载 ml-1m 数据，确保文件路径正确
    filepath = "ml-1m/ratings.dat"
    df = load_ml1m(filepath)
    num_users = df['UserID'].max() + 1
    num_items = df['MovieID'].max() + 1
    print("用户总数：", num_users, "电影总数：", num_items)

    # 划分训练和测试数据
    train_np, test_np = build_user_item_matrix(df, num_users, num_items, split_ratio=0.8)

    # 构造归一化邻接矩阵，并确保移动到GPU上
    norm_adj = create_norm_adj_matrix(train_np)

    # 构造 BPR 训练数据集，并将其用于 DataLoader（数据仍为CPU，采样时转换到GPU）
    train_dataset = BPRDataset(train_np)

    # 初始化 LightGCN 模型，并移到GPU上
    embedding_dim = 64
    num_layers = 3
    model = LightGCN(num_users, num_items, embedding_dim, num_layers, dropout=0.1).cuda()

    # 训练模型
    # train_lightgcn(model, norm_adj, train_dataset, num_epochs=50, batch_size=1024, lr=0.001)
    model.load_state_dict(torch.load('lightgcn_model.pth'))
    # 设置模型为评估模式
    model.eval()
    print("模型已加载。")
    # 评估模型，计算 Recall@20 与 NDCG@20
    evaluate_lightgcn(model, norm_adj, train_np, test_np, topk=20)
    torch.save(model.state_dict(), 'lightgcn_model.pth')
    print("模型已保存。")