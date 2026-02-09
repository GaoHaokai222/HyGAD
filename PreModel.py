from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gat import GAT
from .gin import GIN
from model.utils import create_norm
from torch_geometric.utils import dropout_edge,dropout_path
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils import to_dense_adj,dense_to_sparse
# from torch_geometric.utils import from_dense


import sys
from model.models.basic_nn import GCN
# from pygod.models import GCN # HyGAD use this
from model.models.hgnn import HyperGNN, HGNN, trans2hyper, Twins_GCN, replace_gcn_with_hypergraph

from model.models.drop_graph import dropout_subgraph
from E_high import compute_E_high,compute_G_ano,compute_G_ano_by_dual_hypergraph

from collections import defaultdict

#导入余弦相似度
from torch.nn.functional import cosine_similarity
import copy
import numpy as np

import torch.nn as nn

import torch
import torch.nn as nn

# class SimpleMLP(nn.Module):
#     def __init__(self, in_dim, num_hidden, out_dim, num_layers=2, dropout=0.2):
#         super(SimpleMLP, self).__init__()
#         self.num_layers = num_layers
#         self.out_channels = out_dim
#         self.dropout_p = dropout

#         layers = []
#         if num_layers < 1:
#             raise ValueError("num_layers 必须 ≥ 1")

#         if num_layers == 1:
#             # 直接输入到输出
#             layers.append(nn.Linear(in_dim, out_dim))
#         else:
#             # 第一层
#             layers.append(nn.Linear(in_dim, num_hidden))
#             layers.append(nn.PReLU())
#             layers.append(nn.Dropout(dropout))
#             # 中间隐藏层（如果 > 2 层）
#             for _ in range(num_layers - 2):
#                 layers.append(nn.Linear(num_hidden, num_hidden))
#                 layers.append(nn.PReLU())
#                 layers.append(nn.Dropout(dropout))
#             # 最后一层
#             layers.append(nn.Linear(num_hidden, out_dim))

#         self.model = nn.Sequential(*layers)

#     def forward(self, x, edge_index=None, batch=None, return_hidden=False):
#         return self.model(x)

#     def reset_parameters(self):
#         for m in self.children():  # 只遍历直接子模块
#             if hasattr(m, 'reset_parameters'):
#                 m.reset_parameters()

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, dropout=0.2, num_layers=2):
        super(SimpleMLP, self).__init__()
        self.num_layers = num_layers
        self.out_channels = out_dim

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim).cuda())
        else:
            layers.append(nn.Linear(in_dim, num_hidden).cuda())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(num_hidden, num_hidden).cuda())
                layers.append(nn.PReLU().cuda())
                layers.append(nn.Dropout(dropout).cuda())
            layers.append(nn.Linear(num_hidden, out_dim).cuda())

        # ✅ 用 ModuleList 注册
        self.layers = nn.ModuleList(layers).cuda()

    def forward(self, x, edge_index=None, batch=None, return_hidden=False):
        for layer in self.layers:
            x = layer(x)  # ✅ 直接调用，无需 isinstance 判断
        return x

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


def add_zero(data):
    device = data.x.device
    x, edge_index = data.x, data.edge_index

    zero_feas = torch.zeros((data.batch_size, x.size(1)), dtype=torch.float, device=device)
    x = torch.cat((zero_feas, x[data.batch_size:, :], x[:data.batch_size, :]), dim=0)

    cur_num, add_num = data.n_id.size(0), data.batch_size
    add_loop = torch.tensor(np.arange(cur_num, cur_num + add_num), dtype=torch.long, device=device).repeat(2,1)
    edge_index = torch.cat((edge_index, add_loop), dim=1)

    data.x, data.edge_index = x, edge_index

    return data

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

    # 构建出边和入边的映射（节点索引 -> 边索引列表，即超节点索引）
    out_edge_map = defaultdict(list)
    in_edge_map = defaultdict(list)

    for idx, (src, tgt) in enumerate(zip(rows.tolist(), cols.tolist())):
        out_edge_map[src].append(idx)  # 出边：从 src 发出的是第 idx 条边
        in_edge_map[tgt].append(idx)   # 入边：到达 tgt 的是第 idx 条边
    
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
    # X_prime = (X_mean) * edge_degree_sum.unsqueeze(-1)  # (m, d), 67.05
    X_prime = X_mean

    
    return H_dual, D_e, D_v, L_dual, X_prime, out_edge_map, in_edge_map


def conbine_hyper_neighbor(out_edge_map, in_edge_map):
    # 初始化目标 map
    node_to_edge_map = defaultdict(list)

    # 添加出边
    for node, edges in out_edge_map.items():
        node_to_edge_map[node].extend(edges)

    # 添加入边
    for node, edges in in_edge_map.items():
        node_to_edge_map[node].extend(edges)

    # 如果你希望边索引不重复（如无向图场景），可以去重：
    for node in node_to_edge_map:
        node_to_edge_map[node] = list(set(node_to_edge_map[node]))

    return node_to_edge_map

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True,aggr='sum') -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            out_dim=int(out_dim),
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "mlp":
        # * just for decoder 
        # mod = nn.Sequential(
        #     nn.Linear(in_dim, num_hidden),
        #     nn.PReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(num_hidden, out_dim)
        # )
        mod = SimpleMLP(in_dim, num_hidden, out_dim, dropout=dropout, num_layers=num_layers)
    elif m_type == "gcn":
        mod= GCN(
                in_channels=int(in_dim),
                hidden_channels=int(num_hidden),
                num_layers=num_layers,
                out_channels=int(out_dim),
                dropout=dropout,
                act=activation,
                aggr=aggr)
    elif m_type == "simple_gcn":
        # mod = Twins_GCN(input_dim=in_dim, hidden_dim=int(num_hidden), out_dim=int(out_dim) , num_gc_layers=1)
        mod = Twins_GCN(input_dim=in_dim, hidden_dim=int(num_hidden), out_dim=int(out_dim) , num_gc_layers=num_layers)
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class SigmoidDifferenceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat, x):
        diff = torch.abs(x_hat - x)
        loss = 1 / (1 + torch.exp(-diff))
        return loss.mean()


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            encoder_num_layers: int,
            attr_decoder_num_layers: int,
            struct_decoder_num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            attr_decoder_type: str = "gat",
            struct_decoder_type:str='gat',
            loss_fn: str = "sce",
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            drop_edge_rate: float = 0.0,
            drop_path_rate: float=0.0,
            predict_all_node: bool=False,
            predict_all_edge: float=0,
            drop_path_length:int=3,
            walks_per_node:int=3,
            select_gano_num:int=0,
            sano_weight:float=0.0
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._attr_decoder_type = attr_decoder_type
        self._struct_decoder_type=struct_decoder_type
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        self._drop_edge_rate = drop_edge_rate
        self._drop_path_rate = drop_path_rate
        self.predict_all_node=predict_all_node
        self.predict_all_edge=predict_all_edge
        self.drop_path_length=drop_path_length

        self.walks_per_node=walks_per_node
        self.neg_s=None

        self.select_gano_num=select_gano_num
        self.sano_weight=sano_weight

        self.sano_weight=1 # 0?
        self.hyper_weight=1
        # self.hyper_weight=0

        if self.sano_weight==-1:
            self.sano_weight=1
            self.aano_weight=0
        else:
            self.aano_weight=1

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        attr_dec_num_hidden = num_hidden // nhead_out if attr_decoder_type in ("gat", "dotgat") else num_hidden 
        struct_dec_num_hidden = num_hidden // nhead_out if struct_decoder_type in ("gat", "dotgat") else num_hidden 


        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=encoder_num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        # self.encoder = Twins_GCN(input_dim=in_dim, hidden_dim=enc_num_hidden, out_dim=enc_num_hidden , num_gc_layers=encoder_num_layers)
        # build decoder for attribute prediction
        self.attr_decoder = setup_module(
            m_type=attr_decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=attr_dec_num_hidden,
            out_dim=in_dim,
            num_layers=attr_decoder_num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=False,
        )
        self.struct_decoder = setup_module(
            m_type=struct_decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=struct_dec_num_hidden,
            out_dim=in_dim,
            num_layers=struct_decoder_num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=False,
        )
        # enc_num_hidden=256
        # encoder_num_layers=1
        # self.encoder = Twins_GCN(input_dim=in_dim, hidden_dim=enc_num_hidden, out_dim=enc_num_hidden , num_gc_layers=encoder_num_layers)
        # self.attr_decoder = Twins_GCN(input_dim=dec_in_dim, hidden_dim=attr_dec_num_hidden, out_dim=in_dim , num_gc_layers=attr_decoder_num_layers)
        # self.struct_decoder = Twins_GCN(input_dim=dec_in_dim, hidden_dim=struct_dec_num_hidden, out_dim=in_dim , num_gc_layers=struct_decoder_num_layers)
        # self.hyper_encoder = HyperGNN(input_dim=in_dim, input_dim_edge=in_dim, hidden_dim=enc_num_hidden, out_dim=enc_num_hidden , num_gc_layers=encoder_num_layers)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.hyper_encoder = HGNN(input_dim=in_dim, hidden_dim=enc_num_hidden, out_dim=enc_num_hidden , num_gc_layers=encoder_num_layers)
        # self.hyper_encoder = copy.deepcopy(self.encoder)
        # replace_gcn_with_hypergraph(self.hyper_encoder)
        self.enc_num_hidden = enc_num_hidden
        self.attr_hyper_fusion = nn.Sequential(nn.Linear(self.enc_num_hidden, self.enc_num_hidden), 
                                               nn.ReLU(inplace=True),
                                               nn.Linear(self.enc_num_hidden, self.enc_num_hidden))
        self.struct_hyper_fusion = nn.Sequential(nn.Linear(self.enc_num_hidden, self.enc_num_hidden), 
                                               nn.ReLU(inplace=True),
                                               nn.Linear(self.enc_num_hidden, self.enc_num_hidden))
        self.hyper_decoder = HyperGNN(input_dim=dec_in_dim, input_dim_edge=dec_in_dim, hidden_dim=struct_dec_num_hidden,out_dim=in_dim, num_gc_layers=struct_decoder_num_layers)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.concat_num = 1
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim*self.concat_num, dec_in_dim, bias=False)


        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.sig_criterion = self.setup_loss_fn("sigmoid", alpha_l)

        #备忘录
        self.subgraph_cache = {}  # 格式: {center_node: {'best_sub_x': ..., 'best_sub_adj': ..., 'sub_node_indices': ...}}
        # 可配置缓存有效期（如每多少epoch清空一次，避免缓存过大）
        self.cache_epoch_interval = 10  # 每10个epoch清空一次缓存
        self.current_epoch = 0  # 记录当前epoch


    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sigmoid":
            criterion = SigmoidDifferenceLoss()
            self.criterion = criterion
        else:
            raise NotImplementedError
        return criterion
    
    def get_global_subgraph_nodes(self, x, edge_index, center_node):
        """获取中心节点的所有可达节点（全局索引，保持0~N-1范围）"""
        center = center_node.item()
        adj_list = {}
        for u, v in edge_index.t().tolist():
            if u not in adj_list:
                adj_list[u] = []
            adj_list[u].append(v)
            if v not in adj_list:
                adj_list[v] = []
            adj_list[v].append(u)

        # BFS获取所有可达节点（返回全局索引）
        visited = set()
        queue = [center]
        visited.add(center)
        while queue:
            node = queue.pop(0)
            if node in adj_list:
                for neighbor in adj_list[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return torch.tensor(list(visited), device=x.device)  # 全局节点索引
    
    def compute_model_residual_scores(self, x, edge_index, num_nodes):
        """
        核心函数1：利用模型自身预测残差计算节点异常分数
        残差越大 → 模型对该节点预测越不准 → 异常可能性越高
        """
        with torch.no_grad():  # 不参与梯度更新，仅用于评估
            # 1. 原始数据前向传播（复用模型编码器）
            use_edge = add_self_loops(edge_index)[0]
            enc_rep = self.encoder(x, use_edge)
            rep = self.encoder_to_decoder(enc_rep)
            
            # 2. 属性预测残差（节点特征重构误差）
            attr_pred = self.attr_decoder(rep) if self._attr_decoder_type in ("mlp", "linear") else self.attr_decoder(rep, use_edge)
            node_attr_residual = torch.norm(x - attr_pred, dim=1)  # [N]：每个节点的特征残差
            node_attr_residual = node_attr_residual / (node_attr_residual.max() + 1e-6)  # 归一化
            
            # 3. 结构预测残差（边重构误差）
            struct_feat = self.struct_decoder(rep) if self._struct_decoder_type in ("mlp", "linear") else self.struct_decoder(rep, use_edge)
            struct_pred = struct_feat @ struct_feat.T  # [N,N]：预测邻接矩阵
            struct_gt = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # [N,N]：真实邻接矩阵
            node_struct_residual = torch.mean(torch.abs(struct_gt - struct_pred), dim=1)  # [N]：每个节点的边残差
            node_struct_residual = node_struct_residual / (node_struct_residual.max() + 1e-6)  # 归一化
            
            # 4. 融合残差作为异常分数（特征残差权重更高）
            return 0.6 * node_attr_residual + 0.4 * node_struct_residual

    def get_anomaly_centers(self, x, edge_index, num_nodes):
        """核心函数2：基于模型残差筛选异常节点及其邻居作为扰动中心"""
        # 1. 计算节点异常分数
        anomaly_scores = self.compute_model_residual_scores(x, edge_index, num_nodes)
        
        # 2. 筛选前5%高异常节点
        top_k = max(1, int(num_nodes * 0.05))
        anomaly_nodes = torch.argsort(anomaly_scores, descending=True)[:top_k]
        
        # 3. 扩展异常节点的1-hop邻居
        adj_list = {u: [] for u in range(num_nodes)}
        for u, v in edge_index.t().tolist():
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        anomaly_neighbors = set()
        for node in anomaly_nodes.tolist():
            anomaly_neighbors.update(adj_list[node])
        
        # 4. 合并去重并控制数量（最多20%节点）
        candidates = list(set(anomaly_nodes.tolist()) | anomaly_neighbors)
        candidates = torch.tensor(candidates, device=x.device)
        
        if len(candidates) > int(num_nodes * 0.2):
            candidates = candidates[torch.randperm(len(candidates))[:int(num_nodes * 0.2)]]
        
        return candidates if len(candidates) > 0 else torch.randperm(num_nodes)[:max(1, int(num_nodes * 0.1))]
    
    def get_2hop_subgraph(self, x, edge_index, center_node):
        """严格获取2-hop范围内的子图节点"""
        center = center_node.item()
        edge_t = edge_index.t().tolist()
        adj_list = {}
        for u, v in edge_t:
            if u not in adj_list:
                adj_list[u] = set()
            adj_list[u].add(v)
            if v not in adj_list:
                adj_list[v] = set()
            adj_list[v].add(u)

        # 1-hop邻居
        hop1 = adj_list.get(center, set())
        # 2-hop邻居（1-hop邻居的邻居，但排除中心和1-hop本身）
        hop2 = set()
        for node in hop1:
            hop2.update(adj_list.get(node, set()))
        hop2 = hop2 - {center} - hop1  # 去重

        # 合并：中心 + 1-hop + 2-hop
        all_nodes = {center} | hop1 | hop2
        return torch.tensor(list(all_nodes), device=x.device)

    
    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        keep_nodes = perm[num_mask_nodes: ]

        # print('_replace_rate',self._replace_rate)
        if self._replace_rate > 0 and int(self._replace_rate * num_mask_nodes)>0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def forward(self, x, edge_index,batch, denoise=False):
        # ---- attribute and edge reconstruction ----
        if denoise:
            loss,final_edge_mask_rate = self.ablation_denoise(x, edge_index,batch)
        else:
            loss,final_edge_mask_rate = self.mask_attr_prediction(x, edge_index,batch)
        loss_item = {"loss": loss.item()}
        return loss, loss_item,final_edge_mask_rate

    def intersection_edge(self,edge_index_1, edge_index_2,max_num_nodes): 
        s1=to_dense_adj(edge_index_1,max_num_nodes=max_num_nodes)[0]
        s2=to_dense_adj(edge_index_2,max_num_nodes=max_num_nodes)[0]
        intersection_s=torch.min(s1,s2)

        intersection_edge_index,_=dense_to_sparse(intersection_s)
        # print('intersection_edge_index',intersection_edge_index)

        return intersection_edge_index,intersection_s,


    def mask_attr_prediction(self, x, edge_index, batch):
        # num_nodes=x.size()[0]

        # # mask edge to reduce struct uncertainty
        # _mask_rate=self._mask_rate
        # dence_edge_index=to_dense_adj(edge_index,max_num_nodes=num_nodes)[0]
        # # use_e_high=False

        # if self.select_gano_num:

        #     G_ano_init=float('inf')
        #     for j in range(self.select_gano_num):
        #         use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, _mask_rate)
        #         a_ano,s_ano, hyper_ano=compute_G_ano_by_dual_hypergraph(dence_edge_index,use_x)
        #         G_ano=a_ano*self.aano_weight+s_ano*self.sano_weight+hyper_ano*self.hyper_weight
        #       #print('G_ano',G_ano)    
        #         if G_ano<G_ano_init:
        #             use_x_select=use_x
        #             mask_nodes_select=mask_nodes
        #             keep_nodes_select=keep_nodes
        #             G_ano_init=G_ano
        #   #print('final G_ano',G_ano_init)
        #     use_x=use_x_select
        #     mask_nodes=mask_nodes_select
        #     keep_nodes=keep_nodes_select
        # else:
        #     use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, _mask_rate)

        num_nodes = x.size()[0]
        _mask_rate = self._mask_rate
        dence_edge_index = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        
        # 新增：中心节点配置参数（默认20个，支持数值/百分比）
        self.central_node_count = getattr(self, 'central_node_count', 20)  # 具体数量
        self.central_node_ratio = getattr(self, 'central_node_ratio', None)  # 百分比
        self.use_central_nodes = getattr(self, 'use_central_nodes', False)  # 是否启用中心节点
        
        # 核心：计算实际中心节点数量（优先百分比）
        def get_central_count():
            if self.central_node_ratio is not None:
                ratio = max(0.01, min(0.5, self.central_node_ratio))
                return max(1, int(num_nodes * ratio))
            else:
                return min(self.central_node_count, num_nodes)
        
        # 新增：获取2-hop子图节点
        def get_2hop_subgraph(center):
            center = center.item()
            edge_t = edge_index.t().tolist()
            adj_list = {}
            for u, v in edge_t:
                if u not in adj_list: adj_list[u] = set()
                if v not in adj_list: adj_list[v] = set()
                adj_list[u].add(v)
                adj_list[v].add(u)
            hop1 = adj_list.get(center, set())
            hop2 = set()
            for node in hop1:
                hop2.update(adj_list.get(node, set()))
            hop2 = hop2 - {center} - hop1
            return torch.tensor(list({center} | hop1 | hop2), device=x.device)
        
        # 阶段1：如果启用中心节点，先对子图进行去噪
        if self.use_central_nodes and self.select_gano_num:
            central_count = get_central_count()
            central_nodes = torch.randperm(num_nodes, device=x.device)[:central_count]
            perturbed_x = x.clone()
            perturbed_adj = dence_edge_index.clone()
            
            for center in central_nodes:
                sub_nodes = get_2hop_subgraph(center)
                if len(sub_nodes) < 2: continue
                sub_x = perturbed_x[sub_nodes]
                sub_edge_mask = torch.zeros_like(perturbed_adj, dtype=torch.bool)
                sub_edge_mask[sub_nodes, :][:, sub_nodes] = True
                
                # 子图内贪心扰动（保留原有逻辑）
                best_g_ano = float('inf')
                best_sub_x, best_sub_adj = None, None
                for _ in range(self.select_gano_num):
                    perturb_sub_x, _ = self.encoding_mask_noise(sub_x.clone(), _mask_rate)
                    perturb_sub_adj = perturbed_adj.clone()
                    if hasattr(self, '_drop_edge_rate') and self._drop_edge_rate > 0:
                        edge_mask = (torch.rand_like(perturb_sub_adj) > self._drop_edge_rate) & sub_edge_mask
                        perturb_sub_adj = perturb_sub_adj * edge_mask.float()
                    
                    temp_x = perturbed_x.clone()
                    temp_x[sub_nodes] = perturb_sub_x
                    a_ano, s_ano, hyper_ano = compute_G_ano_by_dual_hypergraph(perturb_sub_adj, temp_x)
                    current_g_ano = a_ano * self.aano_weight + s_ano * self.sano_weight + hyper_ano * self.hyper_weight
                    
                    if current_g_ano < best_g_ano:
                        best_g_ano = current_g_ano
                        best_sub_x = perturb_sub_x.detach().clone()
                        best_sub_adj = perturb_sub_adj.detach().clone()
                
                perturbed_x[sub_nodes] = best_sub_x.clone()
                perturbed_adj = best_sub_adj.clone()
            
            # 子图去噪后，用原图的edge_index继续处理
            use_edge_index = edge_index
        else:
            # 不启用中心节点时，直接使用原始数据
            perturbed_x = x.clone()
            perturbed_adj = dence_edge_index.clone()
            use_edge_index = edge_index
        
        # 阶段2：无论是否启用中心节点，都必须用encoding_mask_noise处理整张图（核心要求）
        if self.select_gano_num:
            # 原有贪心选择逻辑（作用于阶段1的结果或原始数据）
            G_ano_init = float('inf')
            for j in range(self.select_gano_num):
                # 关键：始终调用encoding_mask_noise处理整张图
                use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(perturbed_x, _mask_rate)
                a_ano, s_ano, hyper_ano = compute_G_ano_by_dual_hypergraph(perturbed_adj, use_x)
                G_ano = a_ano * self.aano_weight + s_ano * self.sano_weight + hyper_ano * self.hyper_weight
                if G_ano < G_ano_init:
                    use_x_select = use_x
                    mask_nodes_select = mask_nodes
                    keep_nodes_select = keep_nodes
                    G_ano_init = G_ano
            use_x = use_x_select
            mask_nodes = mask_nodes_select
            keep_nodes = keep_nodes_select
        else:
            # 基础逻辑：直接调用encoding_mask_noise（必用）
            use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(perturbed_x, _mask_rate)
            use_edge_index = edge_index


        use_x=use_x.to(torch.float32)
        # print('use_x',use_x.size())
        _drop_path_rate=self._drop_path_rate
        _drop_edge_rate=self._drop_edge_rate


        # mask edge for struct reconstruction
        if _drop_edge_rate > 0:
            # use_mask_edge_edge_index, masked_edge_edges = dropout_edge(edge_index, _drop_edge_rate)
            
            if self.select_gano_num:
                G_ano_init=float('inf')
                for j in range(self.select_gano_num):
                    use_mask_edge_edge_index, masked_edge_edges = dropout_edge(edge_index, _drop_edge_rate)
                    # to_dense_adj(edge_index)[0]
                    a_ano,s_ano,hyper_ano=compute_G_ano_by_dual_hypergraph(to_dense_adj(use_mask_edge_edge_index,max_num_nodes=num_nodes)[0],use_x)
                    G_ano=a_ano*self.aano_weight+s_ano*self.sano_weight+hyper_ano*self.hyper_weight
                  #print('G_ano',G_ano)    
                    if G_ano<G_ano_init:
                        use_mask_edge_edge_index_select=use_mask_edge_edge_index
                        masked_edge_edges_select=masked_edge_edges
                        G_ano_init=G_ano
              #print('final G_ano',G_ano_init)
                use_mask_edge_edge_index=use_mask_edge_edge_index_select
                masked_edge_edges=masked_edge_edges_select
            else:
                use_mask_edge_edge_index, masked_edge_edges = dropout_edge(edge_index, _drop_edge_rate)

            use_mask_edge_edge_index = add_self_loops(use_mask_edge_edge_index)[0]
        else:
            # use_mask_edge_edge_index = edge_index
            use_mask_edge_edge_index = use_edge_index

        # mask path for struct reconstruction
        if _drop_path_rate > 0:
            if self.select_gano_num:
                G_ano_init=float('inf')
                for j in range(self.select_gano_num):
                    use_mask_path_edge_index, masked_path_edges,_= dropout_subgraph(edge_index, p=_drop_path_rate,walk_length=self.drop_path_length,walks_per_node=self.walks_per_node,return_subgraph=False)
                  #print('to_dense_adj(use_mask_path_edge_index)[0],use_x',to_dense_adj(use_mask_path_edge_index,max_num_nodes=num_nodes)[0].size(),use_x.size())
                    a_ano,s_ano, hyper_ano=compute_G_ano_by_dual_hypergraph(to_dense_adj(use_mask_path_edge_index,max_num_nodes=num_nodes)[0],use_x)
                    G_ano=a_ano*self.aano_weight+s_ano*self.sano_weight+hyper_ano*self.hyper_weight
                  #print('G_ano',G_ano)    
                    if G_ano<G_ano_init:
                        use_mask_path_edge_index_select=use_mask_path_edge_index
                        masked_path_edges_select=masked_path_edges
                        G_ano_init=G_ano
              #print('final G_ano',G_ano_init)
                use_mask_path_edge_index=use_mask_path_edge_index_select
                masked_path_edges=masked_path_edges_select
            else:
                use_mask_path_edge_index, masked_path_edges,_= dropout_subgraph(edge_index, p=_drop_path_rate,walk_length=self.drop_path_length,walks_per_node=self.walks_per_node,return_subgraph=False)
            
            use_mask_path_edge_index = add_self_loops(use_mask_path_edge_index)[0]
        else:
            # use_mask_path_edge_index = edge_index
            use_mask_path_edge_index = use_edge_index
            
        # mask edge and path
        use_edge_index,use_s=self.intersection_edge(use_mask_edge_edge_index,use_mask_path_edge_index,num_nodes)

        self.return_hidden=False

        if self.return_hidden==False:
            enc_rep= self.encoder(use_x, use_edge_index, return_hidden=self.return_hidden)
        else:
            enc_rep, all_hidden = self.encoder(use_x, use_edge_index, return_hidden=self.return_hidden)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        #hyper process
        H_dual, D_e, D_v, L_dual, X_prime, out_edge_map, in_edge_map = construct_dual_hypergraph_tensor(to_dense_adj(use_mask_edge_edge_index,max_num_nodes=num_nodes)[0], use_x)
        
        hyperedge_index = torch.stack([H_dual.indices()[1], H_dual.indices()[0]], dim=0)
        # with torch.no_grad():
        #     if self.return_hidden==False:
        #         hyper_enc_rep = self.hyper_encoder(X_prime, hyperedge_index, return_hidden=False)
        #     else:
        #         hyper_enc_rep,_ = self.hyper_encoder(X_prime, hyperedge_index, return_hidden=False)
        if self.return_hidden==False:
            hyper_enc_rep = self.hyper_encoder(X_prime, hyperedge_index, return_hidden=False)
        else:
            hyper_enc_rep,_ = self.hyper_encoder(X_prime, hyperedge_index, return_hidden=False)


        # attr_rep_to_decoder = self.attr_hyper_fusion(H_dual.T @ hyper_enc_rep)
        # attr_rep_to_decoder = self.encoder_to_decoder(torch.cat([enc_rep, attr_rep_to_decoder], dim=1))
        # enc_rep_fused = torch.cat([enc_rep, attr_rep_to_decoder], dim=1)
        enc_rep_fused = enc_rep
        

        # ---- attribute and edge reconstruction ----
        rep = self.encoder_to_decoder(enc_rep_fused)
        loss=0
        final_mask_rate=0

        # # ---- contrasive loss ----
        # # select_nodes = batch.input_id
        # # 节点级的损失用n_id更好
        # # select_nodes = batch.n_id
        # select_nodes = keep_nodes
        # batch_node_fea = rep[select_nodes]
        # node_to_hypernodes = conbine_hyper_neighbor(out_edge_map, in_edge_map)
        # readout_features = []
        # for v in select_nodes.tolist():
        #     hypernode_ids = node_to_hypernodes[v]  # list[int]
        #     if len(hypernode_ids) == 0:
        #         # 如果没有边，设为零向量
        #         pooled = torch.zeros_like(hyper_enc_rep[0])
        #     else:
        #         pooled = hyper_enc_rep[hypernode_ids].mean(dim=0)
        #     readout_features.append(pooled)

        # node_sim_loss = 1-cosine_similarity(batch_node_fea, torch.stack(readout_features, dim=0), dim=-1).mean()
        # # origin_subgraph_pooled = batch_node_fea.mean(dim=0)
        # # hyper_subgraph_pooled = torch.stack(readout_features, dim=0).mean(dim=0)
        # # subgraph_sim_loss = 1-cosine_similarity(origin_subgraph_pooled, hyper_subgraph_pooled, dim=-1).mean()
        # # loss += 0.9*node_sim_loss + 0.1*subgraph_sim_loss
        # # print(node_sim_loss)
        # loss += node_sim_loss

        # # center contra loss
        # center_nodes = batch.input_id
        # center_node_fea = rep[center_nodes]
        # node_to_hypernodes = conbine_hyper_neighbor(out_edge_map, in_edge_map)
        # center_readout_features = []
        # for v in center_nodes.tolist():
        #     hypernode_ids = node_to_hypernodes[v]  # list[int]
        #     if len(hypernode_ids) == 0:
        #         # 如果没有边，设为零向量
        #         pooled = torch.zeros_like(hyper_enc_rep[0])
        #     else:
        #         pooled = hyper_enc_rep[hypernode_ids].mean(dim=0)
        #     center_readout_features.append(pooled)

        # node_sim_loss = 1-cosine_similarity(batch_node_fea, torch.stack(readout_features, dim=0), dim=-1).mean()
        # center_node_sim_loss = 1-cosine_similarity(center_node_fea, torch.stack(center_readout_features, dim=0), dim=-1).mean()
        # alpha = 1
        # beta = 0.0
        # loss += alpha*node_sim_loss + beta*center_node_sim_loss



        # ---- Hyper Loss Calculation ----
        # hyper_recon,_ = self.hyper_decoder(hyper_enc_rep, hyperedge_index) # TODO:超图中间特征的mlp
        # loss += self.criterion(hyper_recon @ hyper_recon.T, L_dual)
        # # loss += self.sig_criterion(hyper_recon @ hyper_recon.T, L_dual)
        # # loss += self.criterion(hyper_recon, X_prime)


        # ---- attribute reconstruction ----
        if _mask_rate>0:
            # if self._attr_decoder_type not in ("mlp", "linear"):
            #     # * remask, re-mask
            #     rep[mask_nodes] = 0
            if self._attr_decoder_type in ("mlp", "linear") :
                attr_recon = self.attr_decoder(rep)
            else:
                attr_recon = self.attr_decoder(rep, use_edge_index)

            use_mask_loss=False
            if use_mask_loss:
                x_init = x[mask_nodes]
                x_rec = attr_recon[mask_nodes]
            else:
                x_init = x
                x_rec = attr_recon

            loss += self.criterion(x_rec, x_init)
        # ---- edge reconstruction ----
        if _drop_edge_rate>0 or _drop_path_rate>0 :
            if self._struct_decoder_type in ("mlp", "linear") :
                h_recon = self.struct_decoder(rep)
            else:
                h_recon = self.struct_decoder(rep, use_edge_index)      
        
            struct_recon = h_recon @ h_recon.T

            s_init = to_dense_adj(edge_index,max_num_nodes=num_nodes)[0]

            mask_edge_num=(use_s==0) & (s_init==1)
            final_mask_rate=mask_edge_num.sum()/edge_index.size()[1]

            if self.predict_all_edge:
                if  self.neg_s==None:
                    neg_rate=edge_index.size()[1]/(s_init.size()[0]**2)*self.predict_all_edge
                    self.neg_s=torch.rand(s_init.size()) <neg_rate
                   
                s_rec = torch.where((((use_s==0) & (s_init==1))|(self.neg_s).to(use_s.device)),struct_recon,s_init)
                self.neg_s=None
            else:
                s_rec = torch.where((use_s==0) & (s_init==1),struct_recon,s_init)

            if _drop_edge_rate>0:
                loss += self.sig_criterion(s_rec, s_init)
            if _drop_path_rate>0:
                loss += self.criterion(s_rec, s_init)
        
        return loss,final_mask_rate

    def ablation_denoise(self, x, edge_index, batch):
        num_nodes=x.size()[0]

        # mask edge to reduce struct uncertainty
        _mask_rate=self._mask_rate
        dence_edge_index=to_dense_adj(edge_index,max_num_nodes=num_nodes)[0]
        # use_e_high=False
        _drop_path_rate=self._drop_path_rate
        _drop_edge_rate=self._drop_edge_rate

        use_x=x.to(torch.float32)
        use_mask_edge_edge_index = edge_index
        use_mask_path_edge_index = edge_index
            
        # mask edge and path
        use_edge_index,use_s=self.intersection_edge(use_mask_edge_edge_index,use_mask_path_edge_index,num_nodes)

        self.return_hidden=False

        if self.return_hidden==False:
            enc_rep= self.encoder(use_x, use_edge_index, return_hidden=self.return_hidden)
        else:
            enc_rep, all_hidden = self.encoder(use_x, use_edge_index, return_hidden=self.return_hidden)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        #hyper process
        H_dual, D_e, D_v, L_dual, X_prime, out_edge_map, in_edge_map = construct_dual_hypergraph_tensor(to_dense_adj(use_mask_edge_edge_index,max_num_nodes=num_nodes)[0], use_x)
        hyperedge_index = torch.stack([H_dual.indices()[1], H_dual.indices()[0]], dim=0)
        if self.return_hidden==False:
            hyper_enc_rep = self.hyper_encoder(X_prime, hyperedge_index, return_hidden=False)
        else:
            hyper_enc_rep,_ = self.hyper_encoder(X_prime, hyperedge_index, return_hidden=False)

        enc_rep_fused = enc_rep
        # ---- attribute and edge reconstruction ----
        rep = self.encoder_to_decoder(enc_rep_fused)
        loss=0
        final_mask_rate=0

        # ---- contrasive loss ----
        # select_nodes = batch.input_id
        # 节点级的损失用n_id更好
        # select_nodes = batch.n_id
        # # select_nodes = keep_nodes
        # batch_node_fea = rep[select_nodes]
        # node_to_hypernodes = conbine_hyper_neighbor(out_edge_map, in_edge_map)
        # readout_features = []
        # for v in select_nodes.tolist():
        #     hypernode_ids = node_to_hypernodes[v]  # list[int]
        #     if len(hypernode_ids) == 0:
        #         # 如果没有边，设为零向量
        #         pooled = torch.zeros_like(hyper_enc_rep[0])
        #     else:
        #         pooled = hyper_enc_rep[hypernode_ids].mean(dim=0)
        #     readout_features.append(pooled)

        # node_sim_loss = 1-cosine_similarity(batch_node_fea, torch.stack(readout_features, dim=0), dim=-1).mean()
        # loss += node_sim_loss

        # ---- attribute reconstruction ----
        if _mask_rate>0:
            # if self._attr_decoder_type not in ("mlp", "linear"):
            #     # * remask, re-mask
            #     rep[mask_nodes] = 0
            if self._attr_decoder_type in ("mlp", "linear") :
                attr_recon = self.attr_decoder(rep)
            else:
                attr_recon = self.attr_decoder(rep, use_edge_index)

            x_init = x
            x_rec = attr_recon
            loss += self.criterion(x_rec, x_init)
        # ---- edge reconstruction ----
        if _drop_edge_rate>0 or _drop_path_rate>0 :
            if self._struct_decoder_type in ("mlp", "linear") :
                h_recon = self.struct_decoder(rep)
            else:
                h_recon = self.struct_decoder(rep, use_edge_index)      
        
            struct_recon = h_recon @ h_recon.T

            s_init = to_dense_adj(edge_index,max_num_nodes=num_nodes)[0]

            mask_edge_num=(use_s==0) & (s_init==1)
            final_mask_rate=mask_edge_num.sum()/edge_index.size()[1]

            if self.predict_all_edge:
                if  self.neg_s==None:
                    neg_rate=edge_index.size()[1]/(s_init.size()[0]**2)*self.predict_all_edge
                    self.neg_s=torch.rand(s_init.size()) <neg_rate
                   
                s_rec = torch.where((((use_s==0) & (s_init==1))|(self.neg_s).to(use_s.device)),struct_recon,s_init)
                self.neg_s=None
            else:
                s_rec = torch.where((use_s==0) & (s_init==1),struct_recon,s_init)

            if _drop_edge_rate>0:
                loss += self.sig_criterion(s_rec, s_init)
            if _drop_path_rate>0:
                loss += self.criterion(s_rec, s_init)
        return loss,final_mask_rate

    def embed(self, x, edge_index):
        rep = self.encoder(x, edge_index)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
    
    @torch.no_grad()
    def update_hyper_encoder_with_ema(self, momentum=0.99):
        """
        EMA update: hyper_encoder = momentum * hyper_encoder + (1 - momentum) * encoder
        """
        for param_src, param_tgt in zip(self.encoder.parameters(), self.hyper_encoder.parameters()):
            param_tgt.data.mul_(momentum).add_(param_src.data, alpha=1 - momentum)
