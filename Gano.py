import torch
import torch_geometric.utils as pyg_utils

def compute_E_high(adj_matrix, feat_matrix):
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    feat_tensor = feat_matrix.clone().detach().to(dtype=torch.float32)

    deg_tensor = torch.sum(adj_tensor, dim=1)
    deg_matrix = torch.diag(deg_tensor)

    laplacian_tensor = deg_matrix - adj_tensor # 拉普拉斯矩阵
    numerator = torch.matmul(torch.matmul(feat_tensor.T, laplacian_tensor), feat_tensor) #X^T L X
    denominator = torch.matmul(feat_tensor.T, feat_tensor) # X^T X
    
    S_high = torch.sum(numerator) / torch.sum(denominator)

    return S_high.item()

def compute_G_ano(adj_matrix, feat_matrix):
    a_high = compute_E_high(adj_matrix, feat_matrix)
    deg_matrix = torch.diag(torch.sum(torch.tensor(adj_matrix, dtype=torch.float32), dim=1))
    s_high = compute_E_high(adj_matrix, deg_matrix)

    return a_high, s_high


def construct_dual_hypergraph_tensor(A: torch.Tensor, X: torch.Tensor, device="cuda"):
    """
    输入:
        A: 邻接矩阵 (n x n, torch.Tensor, 支持稠密或稀疏格式)
        X: 节点特征矩阵 (n x d, torch.Tensor)
        device: 计算设备("cpu" 或 "cuda")
    输出:
        H_dual: 对偶超图的关联矩阵 (m x n, 稀疏张量)
        D_e: 对偶超图的节点度矩阵 (m x m, 稀疏张量)
        D_v: 对偶超图的超边度矩阵 (n x n, 稀疏张量)
        L_dual: 对偶超图的拉普拉斯矩阵 (m x m, 稀疏张量)
        X_prime: 对偶超图的超边特征 (m x d, 稠密张量)
    """
    # --- Step 1: 提取原超图的关联矩阵 H ---
    n = A.shape[0]
    
    # 将邻接矩阵转换为稀疏坐标格式
    if A.is_sparse:
        edge_indices = A.coalesce().indices()
    else:
        edge_indices = torch.nonzero(A, as_tuple=False).t()  # (2, m)
    
    m = edge_indices.shape[1]  # 边数（超边数量）
    rows, cols = edge_indices  # 边的起点和终点
    
    # 构建关联矩阵 H (n x m) 的稀疏坐标
    H_row = torch.cat([rows, cols])          # 节点索引
    H_col = torch.cat([torch.arange(m), torch.arange(m)]).to(device)  # 超边索引
    H_data = torch.ones(2 * m, dtype=torch.float32, device=device)
    
    # 创建稀疏张量 H (n x m)
    H = torch.sparse_coo_tensor(
        indices=torch.stack([H_row, H_col]),
        values=H_data,
        size=(n, m),
        device=device
    )
    
    # --- Step 2: 构造对偶超图的关联矩阵 H^* = H^T (m x n) ---
    H_dual = H.t().coalesce()  # 转置并合并重复索引
    
    # --- Step 3: 计算度矩阵 ---
    # 原超图的节点度矩阵 D_v (n x n 对角稀疏矩阵)
    node_degrees = torch.sparse.sum(H, dim=1).to_dense()  # 行和，即节点度数
    D_v_indices = torch.arange(n, device=device).unsqueeze(0).repeat(2, 1)
    D_v = torch.sparse_coo_tensor(
        indices=D_v_indices,
        values=node_degrees,
        size=(n, n),
        device=device
    )
    
    # 对偶超图的节点度矩阵 D_e (m x m 对角稀疏矩阵，每个超边度数为2)
    hyperedge_degrees = 2 * torch.ones(m, device=device)
    D_e_indices = torch.arange(m, device=device).unsqueeze(0).repeat(2, 1)
    D_e = torch.sparse_coo_tensor(
        indices=D_e_indices,
        values=hyperedge_degrees,
        size=(m, m),
        device=device
    )
    
    # --- Step 4: 计算对偶超图的拉普拉斯矩阵 L^* = D_e - H^* D_v^{-1} H^*^T ---
    # 计算 D_v^{-1}（避免除以零，添加小epsilon）
    D_v_inv_values = 1.0 / (node_degrees + 1e-8)
    D_v_inv = torch.sparse_coo_tensor(
        indices=D_v_indices,
        values=D_v_inv_values,
        size=(n, n),
        device=device
    )
    
    # 稀疏矩阵乘法: H^* @ D_v^{-1} @ H^*^T
    term = torch.sparse.mm(H_dual, D_v_inv)
    term = torch.sparse.mm(term, H_dual.t())
    
    # 构造 L_dual = D_e - term
    L_dual = D_e.coalesce() - term.coalesce()
    
    # --- Step 5: 计算对偶超图的超边特征 X' (m x d) ---
    edge_degree_sum = 1/(node_degrees[rows] + node_degrees[cols])  # (m,)

    # 计算超边的特征，使用其连接的顶点的度之和加权特征
    X_mean = (X[rows] + X[cols]) / 2  # (m, d)
    # X_std = torch.sqrt(((X[rows] - X_mean) ** 2 + (X[cols] - X_mean) ** 2) / 2.0 + 1e-8) 
    X_prime = (X_mean) * edge_degree_sum.unsqueeze(-1)  # (m, d), 67.05

    
    return H_dual, D_e, D_v, L_dual, X_prime

def compute_hyper_RQ(features, Laplacian):
    
    numerator = torch.matmul(torch.matmul(features.T, Laplacian), features) #X^T L X
    denominator = torch.matmul(features.T, features) # X^T X
    
    S_high = torch.sum(numerator) / torch.sum(denominator)

    return S_high.item()


def compute_G_ano_by_dual_hypergraph(adj_matrix, feat_matrix):
    a_high, s_high = compute_G_ano(adj_matrix, feat_matrix)
    #对偶超图转化
    H_dual, D_e, D_v, L_dual, X_prime = construct_dual_hypergraph_tensor(adj_matrix, feat_matrix, device=adj_matrix.device)
    # with torch.no_grad():
    #     hpyer_high = compute_hyper_RQ(X_prime.cpu(), L_dual.cpu())
    hpyer_high = compute_hyper_RQ(X_prime, L_dual)
    return a_high, s_high, hpyer_high

    
