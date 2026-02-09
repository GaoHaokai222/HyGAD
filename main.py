
import logging
import numpy as np
from tqdm import tqdm
import torch

import sys
sys.path.append('pygod')
import importlib

import os

from pygod.utils import load_data
from pygod.metrics import eval_roc_auc,eval_average_precision,eval_ndcg,eval_precision_at_k,eval_recall_at_k
from pygod.models import ADANET
from torch_geometric.utils import to_dense_adj,add_remaining_self_loops,add_self_loops
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from model.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from model.models import build_model
from model.god import god_evaluation
from model.mygod import stage2evaluation
import copy

import logging

from torch_geometric.utils import to_networkx, from_networkx, is_undirected
import random

import time
import torch
import psutil
import os
import gc
from memory_profiler import memory_usage
import json

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range



def get_loader(args, graph, batch_size=2000):
    need_loader = {'inj_cora': False, 'inj_cora_s': False, 'inj_amazon': True, 'inj_amazon_s': True, 'weibo': True, 'reddit': False, 'disney': False, 'books': False, 'enron': False, 'inj_flickr': True, 'gen_100':False, 'gen_500':False, 'gen_1000':False, 'gen_time':False}
    # Create NeighborLoader for mini-batch training
    # train_loader = NeighborLoader(
    #     graph,
    #     num_neighbors=[-1],
    #     input_nodes=None,  # default to all nodes
    #     batch_size=2000,
    #     num_workers=0,
    #     shuffle=False
    # )
    train_loader = NeighborLoader(
        graph,
        num_neighbors=[4, int(4/2)],
        # replace=True,
        # disjoint=True,
        batch_size=batch_size)
    return train_loader if need_loader[args.dataset] else None

def inj_anomaly_edges(graph):
    y = graph.y  # 节点标签
    edge_index = graph.edge_index

    # 找出所有正常节点的索引
    normal_nodes = (y == 0).nonzero(as_tuple=True)[0]  # 假设标签0为正常
    normal_nodes = normal_nodes.tolist()

    # 构造当前邻接关系的集合，便于快速查重
    existing_edges = set()
    for u, v in edge_index.t().tolist():
        existing_edges.add((u, v))
        existing_edges.add((v, u))  # 无向图双向都存

    # 枚举所有正常节点之间可能的边
    candidate_edges = []
    for i in range(len(normal_nodes)):
        for j in range(i + 1, len(normal_nodes)):
            u, v = normal_nodes[i], normal_nodes[j]
            if (u, v) not in existing_edges and (v, u) not in existing_edges:
                candidate_edges.append((u, v))

    # 按比例随机采样
    num_to_add = int(args.replace_rate * len(edge_index[0]))  # c% 原边数
    num_to_add = min(num_to_add, len(candidate_edges))  # 避免过多
    new_edges = random.sample(candidate_edges, num_to_add)

    # 添加这些边到 edge_index
    new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    if is_undirected(edge_index):
        new_edges_tensor = torch.cat([new_edges_tensor, new_edges_tensor.flip(0)], dim=1)

    graph.edge_index = torch.cat([edge_index, new_edges_tensor], dim=1)
    return graph

def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, model_name,aggr_f,lr_f, max_epoch_f, alpha_f,dropout_f,loss_f,loss_weight_f,T_f, num_hidden=16,logger=None,use_ssl=False,return_edge_score=False):
    logging.info("start training..")
    # graph = graph.to(device)
    # x = feat.to(device)

    batch_size = 1000 if args.dataset in ['inj_flickr'] else 4000
    train_loader = get_loader(args, graph, batch_size=batch_size)
    epoch_iter =range(max_epoch_f)
    outlier_score_list=[]
    edge_outlier_score_list=[]
    for epoch in tqdm(epoch_iter, desc="Pretraining Epochs"):
        model.train()
        if train_loader is not None:
            for batch in train_loader:
                batch = batch.to(device)
                loss, loss_dict, final_edge_mask_rate = model(batch.x, batch.edge_index,batch, args.ablation_denoise)
                # loss, loss_dict, final_edge_mask_rate = model(batch.x, batch.edge_index)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # model.update_hyper_encoder_with_ema(momentum=0.99)
        else:
            graph = graph.to(device)
            x = feat.to(device)
            graph.n_id = torch.tensor(range(x.size()[0])).to(device)
            graph.input_id = torch.tensor(range(x.size()[0])).to(device)
            loss, loss_dict,final_edge_mask_rate = model(x, graph.edge_index, graph, args.ablation_denoise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # model.update_hyper_encoder_with_ema(momentum=0.99)

        if scheduler is not None:
            scheduler.step()

        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

    # return best_model
    return model,np.array(outlier_score_list),np.array(edge_outlier_score_list)

def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch

    num_hidden = args.num_hidden

    node_encoder_num_layers = args.node_encoder_num_layers
    edge_encoder_num_layers = args.edge_encoder_num_layers
    subgraph_encoder_num_layers = args.subgraph_encoder_num_layers

    attr_decoder_num_layers= args.attr_decoder_num_layers
    struct_decoder_num_layers= args.struct_decoder_num_layers

    attr_encoder_name = args.attr_encoder
    struct_encoder_name = args.struct_encoder
    topology_encoder_name=args.topology_encoder

    attr_decoder_name = args.attr_decoder
    struct_decoder_name= args.struct_decoder

    replace_rate = args.replace_rate
    weight_decay=args.weight_decay
    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr

    model_name=args.model_name
    aggr_f=args.aggr_f
    max_epoch_f = args.max_epoch_f
    lr_f = args.lr_f
    alpha_f= args.alpha_f
    dropout_f=args.dropout_f
    loss_f=args.loss_f
    loss_weight_f=args.loss_weight_f
    T_f=args.T_f

    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    max_pu_epoch=args.max_pu_epoch
    each_pu_epoch=args.each_pu_epoch

    """
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        x : torch.Tensor
            Attribute (feature) of nodes.
        s : torch.Tensor
            Adjacency matrix of the graph.
        edge_index : torch.Tensor
            Edge list of the graph.
    """
    graph=load_data(dataset_name)
    if type(graph) == tuple:
        graph = graph[0]
        graph.y = graph.ano_n_label
    graph.y = (graph.y==args.detect_type).long()
    if args.test_motivation:
        graph = inj_anomaly_edges(graph)

    graph.edge_index=add_remaining_self_loops(graph.edge_index)[0]
    graph.s=to_dense_adj(graph.edge_index)[0]
    num_features=graph.x.size()[1]
    num_classes=4

    args.num_features = num_features

    auc_score_list = []
    auc_prc_score_list = []

    attr_mask,struct_mask=None,None
    pretrain_auc_score_list=[]
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        seed=int(seed)
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{attr_decoder_type}_{struct_decoder_type}")
        else:
            logger = None

        attr_model,struct_model,topology_model = build_model(args)
        attr_model.to(device)
        struct_model.to(device)
        topology_model.to(device)
        # 注意 encoder/decoder 也要搬
        if hasattr(attr_model, "encoder"):
            attr_model.encoder.to(device)
        if hasattr(attr_model, "attr_decoder"):
            attr_model.attr_decoder.to(device)
        if hasattr(attr_model, "struct_decoder"):
            attr_model.struct_decoder.to(device)

        if args.ablation_HRQ:
            attr_model.sano_weight=0
            attr_model.hyper_weight=0

        if args.parameter_on_RQ:
            attr_model.sano_weight=args.lambda_rq
            attr_model.aano_weight=args.lambda_rq
            attr_model.hyper_weight=args.lambda_hrq

        brandnew_encoder = copy.deepcopy(attr_model.encoder).to(device)

        if args.use_ssl:
            attr_remask=None
            struct_remask=None
            print('======== train attr encoder ========')
            if args.detect_type == 1 or args.detect_type == 2:

                optimizer = create_optimizer(optim_type, attr_model, lr, weight_decay)

                if use_scheduler:
                    logging.info("Use schedular")
                    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                    # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                else:
                    scheduler = None
                    
                x = graph.x
                if not load_model:
                    attr_model,attr_outlier_list,_= pretrain(attr_model, graph, x, optimizer, max_epoch, device, scheduler, model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                    attr_model = attr_model.cpu()

                if load_model:
                    logging.info("Loading Model ... ")
                    attr_model.load_state_dict(torch.load("checkpoint.pt"))
                if save_model:
                    logging.info("Saveing Model ...")
                    torch.save(attr_model.state_dict(), "checkpoint.pt")
                
                attr_model = attr_model.to(device)
                attr_model.eval()

            print('======== train struct encoder ========')
            if args.detect_type == 3:
                struct_model.attr_decoder=attr_model.attr_decoder
                struct_model.struct_decoder=attr_model.struct_decoder
                # struct_model.hyper_encoder = attr_model.hyper_encoder
                # struct_model.hyper_decoder = attr_model.hyper_decoder
                # struct_model.attr_hyper_fusion = attr_model.attr_hyper_fusion

                optimizer = create_optimizer(optim_type, struct_model, lr, weight_decay)

                if use_scheduler:
                    logging.info("Use schedular")
                    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                    # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                else:
                    scheduler = None
                    
                x = graph.x
                if not load_model:
                    struct_model,struct_node_outlier_list,struct_outlier_list= pretrain(struct_model, graph, x, optimizer, max_epoch, device, scheduler,  model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                    struct_model = struct_model.cpu()

                if load_model:
                    logging.info("Loading Model ... ")
                    struct_model.load_state_dict(torch.load("checkpoint.pt"))
                if save_model:
                    logging.info("Saveing Model ...")
                    torch.save(struct_model.state_dict(), "checkpoint.pt")
                
                struct_model = struct_model.to(device)
                struct_model.eval()

            print('======== train topology encoder ========')
            if args.use_encoder_num>=3:
                topology_model.attr_decoder=attr_model.attr_decoder
                topology_model.struct_decoder=struct_model.struct_decoder

                # topology_model.hyper_encoder = attr_model.hyper_encoder
                # topology_model.hyper_decoder = attr_model.hyper_decoder
                # topology_model.attr_hyper_fusion = attr_model.attr_hyper_fusion

                optimizer = create_optimizer(optim_type, topology_model, lr, weight_decay)

                if use_scheduler:
                    logging.info("Use schedular")
                    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                    # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                else:
                    scheduler = None
                    
                x = graph.x
                if not load_model:
                    topology_model,topology_node_outlier_list,topology_outlier_list= pretrain(topology_model, graph, x, optimizer, max_epoch, device, scheduler,  model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                    topology_model = topology_model.cpu()

                if load_model:
                    logging.info("Loading Model ... ")
                    topology_model.load_state_dict(torch.load("checkpoint.pt"))
                if save_model:
                    logging.info("Saveing Model ...")
                    torch.save(topology_model.state_dict(), "checkpoint.pt")
                
                topology_model = topology_model.to(device)
                struct_model.eval()

        print('finish one train!')
        pretraind_model = attr_model #if args.detect_type == 1 else struct_model
        if args.ablation_recon or args.detect_type == 2:
            auc_score,auc_prc,ndcg_score,pk_score,rk_score,final_outlier,_= god_evaluation(dataset_name,model_name,attr_encoder_name,struct_encoder_name,topology_encoder_name,attr_decoder_name,struct_decoder_name,attr_model,struct_model,topology_model,graph, graph.x, aggr_f,lr_f, max_epoch, alpha_f,dropout_f,loss_f,loss_weight_f,T_f,args.num_hidden,node_encoder_num_layers,edge_encoder_num_layers,subgraph_encoder_num_layers,attr_decoder_num_layers,struct_decoder_num_layers,use_ssl=args.use_ssl,use_encoder_num=args.use_encoder_num,attention=args.attention,sparse_attention_weight=args.sparse_attention_weight,theta=args.theta,eta=args.eta)
            output_path = "struct_reslult.json"
            record = {
                'dataset': dataset_name,
                'seed': seed,
                'auc': auc_score,
                'ap': auc_prc,
                'result': to_serializable(final_outlier)
            }
            with open(output_path, 'a', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        else:
            auc_score, precision, recall, auc_prc = stage2evaluation(pretraind_model.encoder,pretraind_model.encoder,dataset_name, device, seed,args.num_hidden, args.test_motivation, args.replace_rate, args.ablation_AADA, args.loss_function_ablation, args)
            # auc_score, precision, recall = stage2evaluation(brandnew_encoder,brandnew_encoder,dataset_name, device, seed,args.num_hidden)
        auc_score_list.append(auc_score)
        auc_prc_score_list.append(auc_prc)

        if logger is not None:
            logger.finish()

    final_auc, final_auc_std = np.mean(auc_score_list), np.std(auc_score_list)
    final_auc_prc, final_auc_prc_std = np.mean(auc_prc_score_list), np.std(auc_prc_score_list)

    print(f"# final_auc: {final_auc*100:.2f}±{final_auc_std*100:.2f}")
    print(f"# final_auc_prc: {final_auc_prc*100:.2f}±{final_auc_prc_std*100:.2f}")
    return auc_score_list, auc_prc_score_list, final_auc, final_auc_std, final_auc_prc, final_auc_prc_std


def run_main():
    main(args)

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()

    args_copy = copy.deepcopy(args)

    
        
    if args.use_cfg:
        args = load_best_configs(args, "config_ada-gad.yml")
        args_copy = load_best_configs(args_copy, "config_ada-gad-origin.yml")#origin_setting

    best_per_dataset = {
        'inj_cora': {'encoder': 'simple_gcn', 'decoder': 'simple_gcn', 'score1': 89.59, 'encoder_layers': 1, 'decoder_layers': 1},
        'inj_cora_s': {'encoder': 'gcn', 'decoder': 'gin', 'score1': 0, 'encoder_layers': 1, 'decoder_layers': 1},
        'inj_amazon': {'encoder': 'mlp', 'decoder': 'gat', 'score1': 96.06, 'encoder_layers': 2, 'decoder_layers': 1},
        'inj_amazon_s': {'encoder': 'gat', 'decoder': 'gcn', 'score1': 0, 'encoder_layers': 1, 'decoder_layers': 1},
        'reddit': {'encoder': 'mlp', 'decoder': 'gin', 'score1': 61.06, 'encoder_layers': 1, 'decoder_layers': 4},
        'books': {'encoder': 'simple_gcn', 'decoder': 'simple_gcn', 'score1': 68.79, 'encoder_layers': 16, 'decoder_layers': 8},
        'enron': {'encoder': 'gat', 'decoder': 'simple_gcn', 'score1': 83.38, 'encoder_layers': 8, 'decoder_layers': 8},
        'disney': {'encoder': 'gcn', 'decoder': 'gin', 'score1': 79.63, 'encoder_layers': 16, 'decoder_layers': 16},
        'gen_100': {'encoder': 'gin', 'decoder': 'gat', 'score1': 0, 'encoder_layers': 1, 'decoder_layers': 1},
        'gen_500': {'encoder': 'gin', 'decoder': 'gat', 'score1': 0, 'encoder_layers': 1, 'decoder_layers': 1},
        'gen_1000': {'encoder': 'gin', 'decoder': 'gat', 'score1': 0, 'encoder_layers': 1, 'decoder_layers': 1},
        'gen_time': {'encoder': 'gin', 'decoder': 'gat', 'score1': 0, 'encoder_layers': 1, 'decoder_layers': 1},
    }

    args.attr_decoder = best_per_dataset[args.dataset]['decoder']
    args.struct_decoder = best_per_dataset[args.dataset]['decoder']
    args.attr_encoder = best_per_dataset[args.dataset]['encoder']
    args.struct_encoder = best_per_dataset[args.dataset]['encoder']
    args.topology_encoder = best_per_dataset[args.dataset]['encoder']
    args.node_encoder_num_layers = best_per_dataset[args.dataset]['encoder_layers']
    args.edge_encoder_num_layers = best_per_dataset[args.dataset]['encoder_layers']
    args.subgraph_encoder_num_layers = best_per_dataset[args.dataset]['encoder_layers']
    args.attr_decoder_num_layers = best_per_dataset[args.dataset]['decoder_layers']
    args.struct_decoder_num_layers = best_per_dataset[args.dataset]['decoder_layers']

    
    # args.max_epoch = 500
    # args.max_epoch_f = 200
    args.num_hidden = 256
    # args.detect_type = 2 # 1 for node, 2 for edge, 3 for both

    args.use_encoder_num = 1
    args.max_epoch_f = 50
    args.use_ssl=True

    #denoise=He_random, recon=recon, gae=GAE, HRQ=HRQ, AADA=pure
    args.ablation_denoise=False 
    args.ablation_recon=False
    args.ablation_gae=False
    args.ablation_HRQ=False # false for HRQ, true for RQ
    args.ablation_AADA=False
    args.efficiency_analysis=False
    args.encoder_decoder_analysis=False
    args.hidden_layers_analysis=False
    args.loss_function_ablation=True
    args.parameter_on_RQ=False

    args.test_motivation=False
    args.replace_rate=0
    
    main(args)

