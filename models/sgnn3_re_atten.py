import math
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.embedding import myBert
from sklearn.decomposition import NMF
from utils.util import my_norm
# from torch_geometric.nn import GCNConv
from torch.distributions.normal import Normal
# from torch_geometric.nn import GATConv
# from torch_geometric.nn import GATv2Conv
import matplotlib.pyplot as plt
import numpy as np
import os

class SGNN(nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, sch_hidden_size, ske_hidden_size):
        super(SGNN, self).__init__()
        self.sch_hidden_size = sch_hidden_size
        self.ske_hidden_size = ske_hidden_size

        self.Bert = myBert()
        self.NodeUpdate = NodeUpdate(input_dim=1024, hidden_dim=256)
        self.GraphSelfAttention = GraphSelfAttention(input_dim=256, num_heads=4)
        self.node_choose = nn.Sequential(nn.Linear(256, 64), nn.Linear(64, 30522), nn.Softmax(), nn.Dropout(p=0.3))
        self.linear = nn.Sequential(nn.Linear(256, self.sch_hidden_size*self.sch_hidden_size), nn.ReLU())

        self.DeterministicAttentionFusion = DeterministicAttentionFusion(d_model = self.sch_hidden_size)

        self.GraphUpdateModule = GraphUpdateModule(input_dim=1024, hidden_dim=512, output_dim=256)
        self.SharedGATMLP = SharedGATMLP(in_dim=256, hidden_dim=128, out_dim=128, heads=4)
        self.SymmetricTemperatureCrossEntropyLoss = SymmetricTemperatureCrossEntropyLoss(temperature=0.1)
        self.GATLayer = GATLayer(in_dim=256, out_dim=256, heads=8, concat=True)
        self.EdgeUpdateLayer = EdgeUpdateLayer(node=self.sch_hidden_size, hidden_dim=128)
        self.NodeUpdateLayer = GCNLayer(128, 128)
        self.Lstm = nn.LSTM(input_size=1024,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            )
        # self.node_choose = nn.Sequential(nn.Linear(128, 64), nn.Dropout(), nn.ReLU(),nn.Linear(64, 8), nn.Dropout(), nn.ReLU(), nn.Softmax())
        self.DIC_conv = GCNLayer(1024, 128)
        self.T_conv = GCNLayer(128, 128)
        self.S_conv = GCNLayer(128, 128)
        self.e = nn.Linear(128, 128)
        self.x = nn.Linear(1024, 128)
        self.reduce = nn.Linear(256, 128)
        # self.schemas_scores = BaseSGNN(8, 128)
        self.q_S = nn.Linear(128, 128)
        self.k_S = nn.Linear(128, 128)
        self.v_S = nn.Linear(128, 128)
        self.layer_norm_S = nn.LayerNorm([128])

        self.q_T = nn.Linear(128, 128)
        self.k_T = nn.Linear(128, 128)
        self.v_T = nn.Linear(128, 128)
        self.layer_norm_T = nn.LayerNorm([128])
        self.Gate_S = nn.Linear(128, 128)
        self.Gate_T = nn.Linear(128, 128)
        self.Fusion = nn.Sequential(nn.Linear(in_features=256, out_features=128),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.3), )
        self.choose = nn.Sequential(nn.Linear(in_features=128, out_features=30522),
                                    nn.Softmax(),
                                    nn.Dropout(p=0.3) )
        self.eps = 1e-12


        self.relu = nn.Sequential(nn.ReLU())
        self.schemas_S_T = SchemasSGNN(128, sch_hidden_size)


    def set_l(self, l):
        self.schemas_sgnn.set_l(l)
        self.skeleton_sgnn.set_l(l)

    def symmetric_normalization(self, A):
        """
        对形状为 (batch_size, N, N) 的邻接矩阵 A 进行对称归一化：
        计算 D^(-1/2) * A * D^(-1/2)
        Args:
            A (torch.Tensor): 形状 (batch_size, N, N) 的邻接矩阵
        Returns:
            torch.Tensor: 归一化后的邻接矩阵 (batch_size, N, N)
        """
        batch_size, N, _ = A.shape
        # 计算度矩阵 D，D_ii = sum_j A_ij
        D = torch.sum(A, dim=2)  # 形状 (batch_size, N)
        # 计算 D^(-1/2)，避免除以 0，加一个极小值 eps=1e-6
        D_inv_sqrt = torch.diag_embed(torch.pow(D + 1e-6, -0.5))  # 形状 (batch_size, N, N)
        # 计算 D^(-1/2) A D^(-1/2)
        A_norm = torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)
        return A_norm

    def forward(self, enity, adj, label, s):
        batch_list = []
        schadj_list = []
        idx_list = []
        embedding_batch_list = []
        idx_target_list = []
        for batch in enity:
            idx_batch_list = []
            if len(embedding_batch_list) <= adj.shape[1]-1:
                node_embedding_list = []
                for i in batch:
                    node_embeddings, idx = self.Bert(i)
                    node_embedding_list.append(node_embeddings)
                    idx_batch_list.append(idx)
                idx_target = torch.Tensor(idx_batch_list).to(torch.int64).to(self.device)
                idx_list.append(idx_target)
                embedding_batch = torch.stack(node_embedding_list, dim=0)
                embedding_batch_list.append(embedding_batch)
            else:
                for i in batch:
                    node_embeddings, idx = self.Bert(i)
                    idx_batch_list.append(idx)
                idx_target = torch.Tensor(idx_batch_list).to(torch.int64).to(self.device)
                idx_list.append(idx_target)
                # idx_target_list.append(idx_target)
        for i in label:
            node_embeddings, idx = self.Bert(i)
            idx_target_list.append(idx)
        idx_target_list = torch.Tensor(idx_target_list).to(torch.int64).to(self.device)

        # target_embeddings, target_idx = self.Bert(label)
        node_embedding = torch.stack(embedding_batch_list, dim=1).squeeze(2)
        # shape[]
        # 学习图全局节点正则化
        adj = torch.tensor(adj, dtype=torch.float32).to(self.device)
        adj_norm = self.symmetric_normalization(adj)
        updated_features = self.NodeUpdate(node_embedding, adj_norm)
        mask_new = torch.zeros(adj.shape[0], adj.shape[1], adj.shape[2]).to(self.device)
        mask_sub = torch.zeros(adj.shape[0], adj.shape[1], adj.shape[2]).to(self.device)
        output = updated_features
        entry_loss_all = 0
        guide_node_loss_all = 0
        guide_edge_loss_all = 0
        score_list_sub_adj = []
        score_list_adj = []
        learn_KL_loss = 0
        for i in range(1, adj.shape[1]):
            # GraphSelfAttention获取相关特征
            X_updated, attn_adj = self.GraphSelfAttention(output[:, :i, :])
            attn_weights = attn_adj.clone().detach()
            # 将前 i*i 个元素设为 1（按行填充）
            mask_new = mask_new.clone()
            mask_new[:, :i+1, :i+1] = 1
            mask_new[:, :i, :i] = 0
            sub_edge_1 = self.linear(X_updated.mean(1))  # (32, 1, 256) -> (32, 1, 100)
            sub_edge = sub_edge_1.reshape(-1, adj.shape[1], adj.shape[1])
            # reshape 成 (32, 10, 10)
            # 只保留 mask 为 1 的值，其他设为 0，作为当前加入节点与图的连通性
            sub_edge = (sub_edge * mask_new).clone()
            sub_edge = self.min_max_normalization(sub_edge)
            # 增量学习每个节点
            node_id = self.node_choose(X_updated.mean(1))
            probs = torch.gather(node_id, dim=-1, index=idx_list[i].unsqueeze(1)).squeeze()
            step_loss = -torch.log(probs + self.eps)
            entry_loss_all = entry_loss_all + step_loss.mean()
            # 计算当前节点与图的邻接边
            # 展平最后两个维度，计算 top k

            # print(i)
            values, indices = torch.topk(attn_weights.reshape(attn_weights.shape[0], -1), int((i*i+1)/2), dim=1)
            # 创建 mask (默认全 0)
            mask = torch.zeros_like(attn_weights).reshape(attn_weights.shape[0], -1)
            # 选定 top k 位置设为 1
            mask = mask.scatter(1, indices, 1)
            # 还原形状
            mask = mask.reshape(attn_weights.shape)
            attn_weights = mask * attn_weights

            # NMF构建
            # A_recon, attn_weights_U, attn_weights_V = self.BatchNMFReconstructor(attn_weights)
            # attn_weights_new_U = self.pad_to_global(attn_weights_U, adj.shape[1])
            # attn_weights_new_V = self.pad_to_global(attn_weights_V, adj.shape[1])
            # attn_weights_new = attn_weights_new_U + attn_weights_new_V

            attn_weights_new = self.pad_to_global(attn_weights, adj.shape[1])

            A_Tn, x_p, U, V = self.DeterministicAttentionFusion(attn_weights_new[:, :i, :i], X_updated.permute(0, 2, 1), output[:, i:i+1, :])
            attn_weights_new_Fusion = self.pad_to_global(A_Tn, attn_weights_new.shape[1])

            # 皮尔逊相关系数计算节点初始邻接矩阵，并通过邻居矩阵进行节点更新
            mask_sub = mask_sub.clone()
            mask_sub.view(-1)[:i * i] = 1
            corr_sch = self.pearson_correlation(node_embedding)
            sub_adj = corr_sch + attn_weights_new_Fusion + sub_edge
            # sub_adj = corr_sch * mask_sub
            output = self.GraphUpdateModule(node_embedding, sub_adj)


            # 数据增强，对学习图（updated_features，adj）和指导图（output，sub_adj）进行操作
            guide_node = output
            guide_edge = sub_adj
            learn_node = updated_features
            learn_edge = self.EdgeUpdateLayer(adj)
            learn_edge_per = learn_edge
            guide_node_enhance, mask_guide = self.exponential_mask(guide_node, 0.1)
            learn_node_enhance, mask_learn = self.exponential_mask(learn_node, 0.1)
            guide = self.SharedGATMLP(guide_node_enhance, guide_edge)
            learn = self.SharedGATMLP(learn_node_enhance, learn_edge)
            guide_node_loss = self.SymmetricTemperatureCrossEntropyLoss(guide, learn)
            guide_edge_loss = self.SymmetricTemperatureCrossEntropyLoss(guide_edge, learn_edge)
            guide_node_loss_all = guide_node_loss_all+ guide_node_loss
            guide_edge_loss_all = guide_edge_loss_all + guide_edge_loss
            if i == 1:
                learn_KL = self.adjacency_kl_loss(learn_edge, adj)
            else:
                learn_KL = self.adjacency_kl_loss(learn_edge, learn_edge_per)
            learn_KL_loss += learn_KL
            # 图相似度计算
            learn_edge_cal = learn_edge[:, :i + 1, :i + 1]
            sub_adj_cal = sub_adj[:, :i + 1, :i + 1]
            adj_cal = adj[:, :i + 1, :i + 1]
            score_sub_adj = self.compute_mcs_score(learn_edge_cal, sub_adj_cal)
            score_adj = self.compute_mcs_score(learn_edge_cal, adj_cal)
            score_list_sub_adj.append(score_sub_adj.mean().item())
            score_list_adj.append(score_adj.mean().item())
            if i == 13:
                print(1)
            #图结构可视化
            # 初始矩阵可视化
            adj_cal_nor = min_max_normalization(adj_cal)
            array = np.array(adj_cal_nor.to('cpu').detach())
            # 使用Matplotlib绘制图形
            plt.imshow(array.squeeze(0), cmap='viridis', interpolation='nearest')
            plt.colorbar()  # 添加颜色条
            # plt.show()
            filepath = './view/STDyn-sgnn3_14/initview{}'.format(s)
            if not os.path.isdir(filepath):
                # 创建文件夹
                os.mkdir(filepath)
            plt.savefig('./view/STDyn-sgnn3_14/initview{}/plot{}.jpg'.format(s, i), format='jpg')
            plt.close()
            # 学习矩阵可视化
            learn_edge_cal_nor = min_max_normalization(learn_edge_cal)
            array = np.array(learn_edge_cal_nor.to('cpu').detach())
            # 使用Matplotlib绘制图形
            plt.imshow(array.squeeze(0), cmap='viridis', interpolation='nearest')
            plt.colorbar()  # 添加颜色条
            # plt.show()
            filepath = './view/STDyn-sgnn3_14/learnview{}'.format(s)
            if not os.path.isdir(filepath):
                # 创建文件夹
                os.mkdir(filepath)
            plt.savefig('./view/STDyn-sgnn3_14/learnview{}/plot{}.jpg'.format(s, i), format='jpg')
            plt.close()
            # 指导矩阵可视化
            sub_adj_nor = min_max_normalization(sub_adj_cal)
            array = np.array(sub_adj_nor.to('cpu').detach())
            # 使用Matplotlib绘制图形
            plt.imshow(array.squeeze(0), cmap='viridis', interpolation='nearest')
            plt.colorbar()  # 添加颜色条
            # plt.show()
            filepath = './view/STDyn-sgnn3_14/guideview{}'.format(s)
            if not os.path.isdir(filepath):
                # 创建文件夹
                os.mkdir(filepath)
            plt.savefig('./view/STDyn-sgnn3_14/guideview{}/plot{}.jpg'.format(s, i), format='jpg')
            plt.close()
            # 注意力权重可视化
            sub_attn_nor = min_max_normalization(attn_weights_new[:, :i + 1, :i + 1])
            array = np.array(sub_attn_nor.to('cpu').detach())
            # 使用Matplotlib绘制图形
            plt.imshow(array.squeeze(0), cmap='viridis', interpolation='nearest')
            plt.colorbar()  # 添加颜色条
            # plt.show()
            filepath = './view/STDyn-sgnn3_14/attnview{}'.format(s)
            if not os.path.isdir(filepath):
                # 创建文件夹
                os.mkdir(filepath)
            plt.savefig('./view/STDyn-sgnn3_14/attnview{}/plot{}.jpg'.format(s, i), format='jpg')
            plt.close()

        # GAT更新学习图边和节点
        learned_node = self.NodeUpdateLayer(learn_edge, learn)
        edge_index = build_fully_connected_edge_index(adj.shape[0], adj.shape[1], self.device)
        # output = self.GATLayer(learn_node, learn_edge)

        P_dist = self.choose(torch.mean(learned_node, dim=1))
        probs = torch.gather(P_dist, dim=-1, index=idx_target_list.unsqueeze(1)).squeeze()
        step_loss = -torch.log(probs + self.eps)
        # entry_loss_pred = entry_loss.mean()
        step_loss_all = step_loss.mean() + 0.1 * entry_loss_all + 0.1 * guide_edge_loss_all + 0.1 * guide_node_loss_all + 0.03 * learn_KL_loss
        # duia1topk_binary_adjacency()

        # accuracy计算
        choose = torch.max(P_dist, dim=1)
        choose_idx = choose.indices
        Target = idx_target_list
        accuracy_gen = torch.sum(torch.eq(Target, choose_idx)) / adj.shape[0]

        mrr, hit_k1, hit_k3, hit_k10 = self.compute_mrr_and_hit(P_dist, idx_target_list)

        return step_loss_all, accuracy_gen, score_list_sub_adj, score_list_adj,  mrr, hit_k1, hit_k3, hit_k10

    def min_max_normalization(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        normalized_tensor = (tensor - min_val) / (max_val - min_val +  self.eps)
        return normalized_tensor

    def compute_mrr_and_hit(self, probabilities, true_indices, k1=1, k3=3, k10=10):
        """
        计算 MRR（Mean Reciprocal Rank） 和 Hit@K
        :param probabilities: Tensor (batch_size, num_candidates)，模型输出的概率分布
        :param true_indices: Tensor (batch_size,)，每个样本的正确答案索引
        :param k: int，计算 Hit@K 时的 K 值
        :return: MRR, Hit@K
        """
        batch_size = probabilities.shape[0]

        # 按概率降序排序，获取排序后的索引
        sorted_indices = torch.argsort(probabilities, dim=1, descending=True)  # (batch_size, num_candidates)

        # 计算 Reciprocal Rank (1/排名)
        ranks = (sorted_indices == true_indices.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1  # 获取排名（从1开始）
        reciprocal_ranks = 1.0 / ranks.float()

        # 计算 Hit@K
        hits1 = (ranks <= k1).float()  # 若排名 ≤ K，则 Hit@K 记为 1，否则为 0
        hits3 = (ranks <= k3).float()  # 若排名 ≤ K，则 Hit@K 记为 1，否则为 0
        hits10 = (ranks <= k10).float()  # 若排名 ≤ K，则 Hit@K 记为 1，否则为 0
        # 计算 MRR 和 Hit@K
        mrr = reciprocal_ranks.mean().item()
        hit_k1 = hits1.mean().item()
        hit_k3 = hits3.mean().item()
        hit_k10 = hits10.mean().item()

        return mrr, hit_k1, hit_k3, hit_k10

    def loss_node(self, output, target):
        return self.schemas_sgnn.loss_node(output, target)

    def accu_node(self, output, target):
        return self.schemas_sgnn.accu_node(output, target)

    def loss_path(self, output, target):
        return self.schemas_sgnn.loss_path(output, target)

    def accu_path(self, output, target):
        return self.schemas_sgnn.accu_path(output, target)

    def pad_to_global(self, tensor, target_size):
        """
        将 (32, i, i) 形状的张量填充为 (32, 10, 10)
        :param tensor: 输入张量，形状 (32, i, i)
        :param target_size: 目标大小（默认10）
        :return: 填充后的张量，形状 (32, 10, 10)
        """
        batch_size, h, w = tensor.shape  # 获取当前大小
        assert h <= target_size and w <= target_size, "i 不能超过 10！"
        # 计算需要填充的量
        pad_h = target_size - h  # 计算高度方向需要填充多少
        pad_w = target_size - w  # 计算宽度方向需要填充多少
        # 使用 F.pad 进行填充 (left, right, top, bottom)
        padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
        return padded_tensor

    def pearson_correlation(self, x):
        """
        计算 (32, 10, 1024) 维度张量的皮尔逊相关系数
        :param x: 输入张量 (batch_size, num_nodes, feature_dim)
        :return: 皮尔逊相关矩阵 (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, feature_dim = x.shape  # (32, 10, 1024)
        # 均值 (batch_size, num_nodes, 1)
        mean_x = x.mean(dim=-1, keepdim=True)
        # 标准差 (batch_size, num_nodes, 1)
        std_x = x.std(dim=-1, keepdim=True, unbiased=False) + 1e-6  # 避免除零
        # 归一化特征 (batch_size, num_nodes, feature_dim)
        x_norm = (x - mean_x) / std_x
        # 计算皮尔逊相关系数 (batch_size, num_nodes, num_nodes)
        pearson_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2)) / feature_dim
        return pearson_matrix

    def exponential_mask(self, X, mask_ratio):
        """
        使用指数分布的权重来决定掩码，较大的特征值被更高概率掩盖
        :param X: 输入张量 (batch, num_nodes, feature_dim) -> (32, i, 256)
        :param mask_ratio: 掩盖比例，默认 10%
        :return: 掩码后的张量 X_masked, 掩码矩阵 mask
        """
        batch_size, num_nodes, feature_dim = X.shape
        # 计算每个特征维度的绝对均值 -> (batch, num_nodes, feature_dim)
        abs_mean = X.abs().mean(dim=0, keepdim=True)  # 计算全局均值 (1, num_nodes, feature_dim)
        # 计算指数权重 -> (1, num_nodes, feature_dim)
        exp_prob = torch.exp(abs_mean)  # 指数变换，使较大值概率高
        exp_prob = exp_prob / exp_prob.sum(dim=-1, keepdim=True)  # 归一化
        # 选择前 10% 概率最高的维度进行掩码
        k = int(feature_dim * mask_ratio)  # 计算要掩盖的维度数量
        _, topk_indices = torch.topk(exp_prob, k, dim=-1, largest=True)  # 选择前 k 个概率最高的索引
        # 构造掩码矩阵 (batch, num_nodes, feature_dim)
        mask = torch.ones(batch_size, num_nodes, feature_dim, device=X.device)
        mask = mask.scatter(-1, topk_indices.expand(batch_size, -1, -1), 0)  # 这些索引的位置设为 0
        # 应用掩码
        X_masked = X * mask
        return X_masked, mask

    def topk_binary_adjacency(self, adj_matrix, k):
        """
        对邻接矩阵 (32,10,10) 进行处理：
        - 取每个矩阵中值最大的 k 个元素，置 1
        - 其余位置置 0
        :param adj_matrix: (batch_size, num_nodes, num_nodes) 的邻接矩阵
        :param k: 需要保留的最大值数量
        :return: 处理后的二值邻接矩阵
        """
        batch_size, num_nodes, _ = adj_matrix.shape
        # 找到每个 (10,10) 矩阵中前 k 个最大值的索引
        topk_values, topk_indices = torch.topk(adj_matrix.view(batch_size, -1), k, dim=1)
        # 创建一个全 0 矩阵
        binary_adj = torch.zeros_like(adj_matrix)
        # 将 top-k 索引对应的值设为 1
        binary_adj.view(batch_size, -1).scatter_(1, topk_indices, 1)
        return binary_adj

    def apply_padding(self, matrix, n, max_size=10):
        """
        对输入矩阵进行填充，使其大小达到 max_size，填充部分为0
        """
        batch_size, current_size, _ = matrix.shape
        if current_size >= max_size:
            return matrix, None
        # 计算填充的数量
        pad_size = max_size - current_size
        padding = (0, pad_size, 0, pad_size)  # (left, right, top, bottom)
        # 填充矩阵
        padded_matrix = F.pad(matrix, padding, value=0)
        return padded_matrix

    def remove_padding(self, matrix, n, max_size=10):
        """
        去掉填充的部分，恢复为原始大小
        """
        return matrix[:, :n, :n]

    def compute_mcs_score(self, adj_matrix1, adj_matrix2):
        """
        计算两个图的 MCS 分数，包括公共边数和公共节点数。
        参数:
        adj_matrix1: (batch_size, i, i) 第一张图的邻接矩阵
        adj_matrix2: (batch_size, j, j) 第二张图的邻接矩阵
        返回:
        mcs_scores: (batch_size,) MCS 分数
        """
        batch_size, i, _ = adj_matrix1.shape
        _, j, _ = adj_matrix2.shape
        # 计算公共边数 (E_MCS)
        common_edges = torch.sum(torch.min(adj_matrix1, adj_matrix2), dim=(1, 2))  # (batch_size,)
        # 计算节点度 (每个节点的边数)
        degree1 = torch.sum(adj_matrix1, dim=-1)  # (batch_size, i)
        degree2 = torch.sum(adj_matrix2, dim=-1)  # (batch_size, j)
        # 计算公共节点数 (V_MCS) - 仅统计度数大于 0 的共同节点
        valid_nodes1 = degree1 > 0  # (batch_size, i) 是否是有效节点
        valid_nodes2 = degree2 > 0  # (batch_size, j) 是否是有效节点
        # 计算最小公共节点数（假设两个图的节点数量不同，需要对齐）
        min_nodes = torch.min(valid_nodes1.sum(dim=-1), valid_nodes2.sum(dim=-1))  # (batch_size,)
        # 计算两个图的边+节点总数
        total_size_1 = torch.sum(adj_matrix1, dim=(1, 2)) + valid_nodes1.sum(dim=-1)
        total_size_2 = torch.sum(adj_matrix2, dim=(1, 2)) + valid_nodes2.sum(dim=-1)
        # 计算 MCS 分数
        mcs_scores = (common_edges + min_nodes) / (torch.max(total_size_1, total_size_2) + 1e-8)
        return mcs_scores

    def adjacency_kl_loss(self, A1, A2):
        """
        计算两张图的 KL 散度损失
        :param A1: 第一张图的邻接矩阵, 形状 (batch_size, i, i)
        :param A2: 第二张图的邻接矩阵, 形状 (batch_size, i, i)
        :return: KL 散度损失值
        """
        epsilon = 1e-10  # 避免数值不稳定
        P = A1 / (A1.sum(dim=-1, keepdim=True) + epsilon)  # 归一化
        Q = A2 / (A2.sum(dim=-1, keepdim=True) + epsilon)  # 归一化
        P = P + epsilon
        Q = Q + epsilon
        kl_loss = F.kl_div(Q.log(), P, reduction='batchmean')  # 计算 KL 散度
        return kl_loss

# 结构性依赖分解和正交矩阵建模
class DeterministicAttentionFusion(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model: 特征维度 d
            r: NMF 核心维度
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def nmf_deterministic(self, A):
        """
        步骤1: 确定性结构依赖建模 (NMF)
        A: 历史邻接矩阵 (B, n-1, n-1)
        返回:
            U: 基矩阵 (B, n-1, r)
            V: 系数矩阵 (B, r, n-1)
        """
        B, N, _ = A.shape
        # 用可微分的NMF迭代更新（确定性规则，无额外可学习参数）
        U = F.softplus(torch.randn(B, N, N, device=A.device))
        V = F.softplus(torch.randn(B, N, N, device=A.device))

        # 固定迭代次数，保证确定性
        for _ in range(10):
            # 更新 V
            U_T = U.transpose(1, 2)
            numerator_V = torch.bmm(U_T, A)
            denominator_V = torch.bmm(torch.bmm(U_T, U), V) + 1e-8
            V = V * (numerator_V / denominator_V)

            # 更新 U
            V_T = V.transpose(1, 2)
            numerator_U = torch.bmm(A, V_T)
            denominator_U = torch.bmm(torch.bmm(U, V), V_T) + 1e-8
            U = U * (numerator_U / denominator_U)

        return U, V

    def orthogonal_projection(self, X_hist):
        """
        步骤2: 确定性特征关联建模 (正交投影矩阵P)
        X_hist: 历史节点特征 (B, d, n-1)
        返回:
            P: 投影矩阵 (B, d, d)
        """
        # P = X^T (XX^T)^{-1} X
        XXT = torch.bmm(X_hist, X_hist.transpose(1, 2))  # (B, d, d)
        # inv_XXT = torch.linalg.inv(XXT + 1e-6 * torch.eye(self.d_model, device=X_hist.device))  # 数值稳定
        # P = torch.bmm(torch.bmm(X_hist.transpose(1, 2), inv_XXT), X_hist)  # (B, d, d)
        return XXT

    def attention_baseline(self, x_p, X_hist):
        """
        基础注意力计算 (可替换为你的注意力实现)
        x_p: 投影后的新增节点特征 (B, d)
        X_hist: 历史节点特征 (B, d, n-1)
        返回:
            attn: 注意力权重 (B, 1, n-1)
        """
        # 缩放点积注意力
        x_p = x_p.unsqueeze(1)  # (B, 1, d)
        attn = torch.bmm(x_p, X_hist) / (self.d_model ** 0.5)  # (B, 1, n-1)
        return attn

    def forward(self, A_guide, X_guide, x_new):
        """
        Args:
            A_guide: 历史邻接矩阵 (B, n-1, n-1)
            X_guide: 历史节点特征 (B, d, n-1)
            x_new: 新增节点特征 (B, d)
        Returns:
            A_Tn: 融合后的注意力权重 (B, 1, n-1)
        """
        B, d, N = X_guide.shape


        # 步骤1: 确定性结构依赖建模 (NMF)
        U, V = self.nmf_deterministic(A_guide)  # U: (B, N, r), V: (B, r, N)

        # 步骤2: 确定性特征关联建模 (正交投影)
        P = self.orthogonal_projection(X_guide)  # (B, d, d)
        x_p = torch.bmm(x_new, P).squeeze(1)  # (B, d)

        # 步骤3: 融合矩阵运算的注意力权重计算
        # 1) 基础语义注意力
        attn_semantic = self.attention_baseline(x_p, X_guide)  # (B, 1, N)

        # 2) 结构-特征确定性关联项: x_p U V X_guide^T
        a = torch.bmm(U, V)
        x_p_UV = torch.bmm(a, x_p.unsqueeze(1).expand(-1, N, -1))  # (B, 1, N)
        # attn_struct = torch.bmm(x_p_UV, X_guide.transpose(1, 2))  # (B, 1, d)
        # 对齐维度到 (B, 1, N)
        attn_struct = torch.bmm(x_p_UV, X_guide)  # (B, 1, N)

        # 3) 哈达玛积融合 + MLP + softmax
        fused = attn_semantic * attn_struct  # 哈达玛积
        # fused_mlp = self.mlp(fused.transpose(1, 2)).transpose(1, 2)  # (B, 1, N)
        A_Tn = F.softmax(fused, dim=-1)  # (B, 1, N)

        return A_Tn, x_p, U, V

# 节点更新
class NodeUpdate(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NodeUpdate, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)  # 线性变换
        self.norm = nn.LayerNorm(hidden_dim)  # 归一化
        self.activation = nn.Sigmoid()  # 激活函数

    def forward(self, X, A):
        """
        X: (batch_size, num_nodes, input_dim) -> (32, 10, 1024)
        A: (batch_size, num_nodes, num_nodes) -> (32, 10, 10)
        """
        batch_size, num_nodes, input_dim = X.shape

        # 计算邻居特征聚合（均值聚合）
        A_sum = torch.sum(A, dim=2, keepdim=True) + 1e-6  # 避免除 0
        A_norm = A / A_sum  # 计算邻居的归一化权重
        X_neighbor = torch.bmm(A_norm, X)  # (batch_size, num_nodes, input_dim)

        # 结合自身特征 + 邻居特征
        X_agg = X + X_neighbor  # 直接相加

        # 通过 MLP 进行特征更新
        X_update = self.fc(X_agg)  # 线性变换
        X_update = self.norm(X_update)  # 归一化
        X_update = self.activation(X_update)  # ReLU 激活

        return X_update

# attention发现邻居节点
class GraphSelfAttention(nn.Module):
    def __init__(self, input_dim=256, num_heads=4):
        super(GraphSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)  # 归一化层
        self.ffn = nn.Sequential(  # 前馈网络
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, X):
        """
        X: (batch_size, num_nodes, input_dim) -> (batch_size, 2~10, 256)
        """
        batch_size, num_nodes, input_dim = X.shape
        # 生成 Key Padding Mask（自动判断有效节点）
        mask = (X.abs().sum(dim=-1) == 0)  # (batch_size, num_nodes)，填充部分全 0
        attn_output, attn_weights = self.attention(X, X, X, key_padding_mask=mask)
        # 残差连接 + 归一化
        X = self.norm(X + attn_output)
        # 通过前馈网络进一步更新特征
        X = self.norm(X + self.ffn(X))
        return X, attn_weights


class GraphUpdateModule(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=256):
        super(GraphUpdateModule, self).__init__()
        # 线性变换
        self.linear1 = nn.Linear(input_dim, hidden_dim)  # 降维
        self.linear2 = nn.Linear(hidden_dim, output_dim)  # 最终降维到 256
        # 注意力机制（GAT-like）
        self.attn_fc = nn.Linear(input_dim, 1)  # 计算注意力分数
        # MLP 进一步处理特征
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(p=0.3),
        )
    def forward(self, X, A):
        """
        X: (batch, num_nodes, input_dim) -> (32, 10, 1024)
        A: (batch, num_nodes, num_nodes) -> (32, 10, 10)
        """
        batch_size, num_nodes, _ = X.shape
        # 归一化邻接矩阵 A_norm = D^(-1/2) A D^(-1/2)
        A_hat = A + torch.eye(num_nodes).to(A.device)  # 自环
        D = A_hat.sum(dim=-1, keepdim=True)  # 度矩阵
        A_norm = A_hat / (D + 1e-6)  # 归一化
        # 计算注意力权重（GAT 机制）
        attn_scores = self.attn_fc(X)  # (32, 10, 1)
        attn_scores = attn_scores.repeat(1, 1, num_nodes)  # (32, 10, 10)
        attn_weights = F.softmax(A_norm * attn_scores, dim=-1)  # 归一化
        # 计算邻居特征聚合
        X_agg = torch.bmm(attn_weights, X)  # (32, 10, 1024)
        # 线性变换 + 激活
        X_hidden = F.relu(self.linear1(X_agg))  # (32, 10, 512)
        # MLP 进一步处理特征
        X_out = self.mlp(X_hidden)  # (32, 10, 256)
        # 跳跃连接
        X_res = self.linear2(F.relu(self.linear1(X)))  # (32, 10, 256)
        X_final = X_out + X_res  # 残差连接
        return X_final  # 输出 (32, 10, 256)

class SharedGATMLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, out_dim=128, heads=4):
        super(SharedGATMLP, self).__init__()
        # self.gat = GATConv(in_dim, hidden_dim, heads=heads, concat=False)
        self.GCNLayer = GCNLayer(in_features=256, out_features=128)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x, edge_matrix):
        """
        x: (batch, nodes, feature_dim) -> (32, i, 256)
        edge_matrix: (batch, 10, 10)
        """
        batch_size, num_nodes, _ = x.shape
        # edge_index = edge_matrix.nonzero(as_tuple=False).t()  # 获取边索引
        # x = x.view(-1, x.shape[-1])  # (32*i, 256)
        # GAT 更新
        x = self.GCNLayer(edge_matrix, x)  # (32*i, hidden_dim)
        # MLP 更新
        x = self.mlp(x)  # (32*i, 128)
        return x.view(batch_size, num_nodes, -1)  # (32, i, 128)

class SymmetricTemperatureCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SymmetricTemperatureCrossEntropyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, learning_graph, guidance_graph):
        """
        learning_graph: (batch, i, 128) -> 学习图的更新节点表示
        guidance_graph: (batch, i, 128) -> 指导图的更新节点表示
        """
        # 计算余弦相似度矩阵 (batch, i, i)
        sim_learning = F.cosine_similarity(learning_graph.unsqueeze(2), learning_graph.unsqueeze(1), dim=-1)
        sim_guidance = F.cosine_similarity(guidance_graph.unsqueeze(2), guidance_graph.unsqueeze(1), dim=-1)

        # 归一化: D^(-1/2) A D^(-1/2)
        def symmetric_norm(sim_matrix):
            degree = sim_matrix.sum(dim=-1, keepdim=True)
            degree_inv_sqrt = torch.pow(degree, -0.5)
            degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
            return degree_inv_sqrt * sim_matrix * degree_inv_sqrt

        P = symmetric_norm(F.softmax(sim_learning / self.temperature, dim=-1))
        P_star = symmetric_norm(F.softmax(sim_guidance / self.temperature, dim=-1))

        # 对称交叉熵损失
        loss = -0.5 * (P_star * torch.log(P + 1e-8) + P * torch.log(P_star + 1e-8)).sum(dim=-1).mean()

        return loss


class EdgeUpdateLayer(nn.Module):
    def __init__(self, node, hidden_dim):
        super(EdgeUpdateLayer, self).__init__()
        self.hidden_dim = hidden_dim
        # 线性层
        self.linear = nn.Linear(node*node, hidden_dim)  # 每个特征输入是1维，输出为hidden_dim
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, node*node),
                                            nn.ReLU())# 输出维度是1，表示更新后的特征
    def forward(self, adj_matrix):
        """
        adj_matrix: (batch_size, n, n) 边关系矩阵，n 是节点数量
        """
        batch_size, n, _ = adj_matrix.shape
        # 将 adj_matrix 展平为 (batch_size * n * n, 1)
        adj_matrix_flat = adj_matrix.view(batch_size, -1)  # (batch_size * n * n, 1)
        # 通过线性层进行边关系更新
        x = F.relu(self.linear(adj_matrix_flat))  # (batch_size * n * n, hidden_dim)
        updated_edges = self.output_layer(x)  # (batch_size * n * n, 1)
        # 恢复为 (batch_size, n, n)
        updated_edges = updated_edges.view(batch_size, n, n)
        return updated_edges

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=8, concat=True):
        super(GATLayer, self).__init__()
        self.heads = heads
        self.concat = concat
        self.out_dim = out_dim // heads  # 每个 head 的输出维度
        self.W = nn.Linear(in_dim, out_dim, bias=False)  # 线性变换
        self.a = nn.Linear(2 * self.out_dim, heads, bias=False)  # 注意力计算
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj, edge_weight=None):
        """
        x: (batch, N, in_dim) -> 节点特征
        adj: (batch, N, N) -> 邻接矩阵
        edge_weight: (batch, N, N) -> 边的权重 (可选)
        """
        B, N, _ = x.shape
        x_transformed = self.W(x).view(B, N, self.heads, self.out_dim)  # (B, N, H, F)
        x_i = x_transformed.unsqueeze(2).repeat(1, 1, N, 1, 1)  # (B, N, N, H, F)
        x_j = x_transformed.unsqueeze(1).repeat(1, N, 1, 1, 1)  # (B, N, N, H, F)
        # 计算注意力分数
        e = self.leaky_relu(self.a(torch.cat([x_i, x_j], dim=-1)).squeeze(-1))  # (B, N, N, H)
        # 处理边权重
        if edge_weight is not None:
            e = e * edge_weight.unsqueeze(-1)  # 让边权重影响注意力
        # Mask 非邻接的部分
        adj_expanded = adj.unsqueeze(-1).expand(-1, -1, -1, self.heads)  # (B, N, N, H)
        # e = e.masked_fill(adj_expanded == 0, float('-inf'))
        # softmax 归一化注意力权重
        # alpha = F.softmax(e, dim=2)  # (B, N, N, H)
        # 计算新的节点表示
        adj_expanded = adj_expanded.permute(0, 1, 3, 2)  # 将 B 变为 (32, 10, 8, 10)
        # 然后进行矩阵乘法，计算每一对节点的特征矩阵
        result = torch.matmul(x_transformed, adj_expanded)  # 结果将会是 (32, 10, 32, 10)
        # 通过 reshape 将结果变为 (32, 10, 256)
        result = result.reshape(32, 10, -1)  # (32, 10, 256)
        # h = torch.einsum('bijkh,bjkh->bikh', adj_expanded, x_transformed)  # (B, N, H, F)
        return F.relu(result)  # 激活函数

def build_fully_connected_edge_index(batch_size, num_nodes, device):
    """ 构造全连接图的 edge_index（适用于 PyG） """
    row = torch.arange(num_nodes, device=device).repeat(num_nodes)
    col = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    edge_index = torch.stack([row, col], dim=0)  # 形状 (2, num_nodes*num_nodes)
    return edge_index



def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

class BaseSGNN(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, limit, hidden_size):
        super(BaseSGNN, self).__init__()
        self.l = limit
        self.Bert = myBert()
        self.hidden_size = hidden_size

        self.gnn = GNN(self.hidden_size)

        self.linear_u_one = nn.Linear(hidden_size, int(0.5 * hidden_size), bias=True)
        self.linear_u_one2 = nn.Linear(int(0.5 * hidden_size), 1, bias=True)
        self.linear_u_two = nn.Linear(hidden_size, int(0.5 * hidden_size), bias=True)
        self.linear_u_two2 = nn.Linear(int(0.5 * hidden_size), 1, bias=True)

        self.multi = Parameter(torch.ones(3))

        self.MARGIN = 0.15
        self.loss_path_function = nn.BCELoss()

    def set_l(self, l):
        self.l = l

    def forward(self, nodes, A):
        hidden = self.gnn(A, nodes)
        scores = self.compute_scores(hidden)
        return scores, hidden

    def loss_node(self, output, target):
        output = output.view(5, -1)
        score_node = torch.mean(output, dim=1, keepdim=True)
        truth = score_node[target: target + 1, :].repeat(5, 1)
        loss = torch.sum((self.MARGIN + score_node - truth).clamp(min=0))
        return loss

    def accu_node(self, output, target):
        score_node = torch.mean(output.view(-1, 5), dim=0, keepdim=True)
        sorted, L = torch.sort(score_node, dim=-1, descending=True)
        target_index = torch.nonzero(L == target)
        rank = (4 - target_index[0, 1]) / 4
        return rank

    def loss_path(self, output, target):
        loss = self.loss_path_function(output, target)
        return loss

    def accu_path(self, output, target):
        p_truth = torch.nonzero(target == 1)
        p_false = torch.nonzero(target == 0)
        no_exist_count = p_false.shape[0]
        truth = output[p_truth[:, 0]].repeat(1, no_exist_count)
        other = output[p_false[:, 0]].T
        zeros = torch.zeros_like(truth)
        ones = torch.ones_like(truth)
        temp = torch.where(truth - other > 0, ones, zeros)
        temp1 = torch.sum(temp, dim=1) / no_exist_count
        accu = torch.mean(temp1, dim=0)
        return accu

    def compute_scores(self, hidden, metric='euclid'):
        # attention on input
        input_a = hidden[:, 0: self.l - 1, :]
        input_b = hidden[:, self.l - 1:, :]
        u_a = F.relu(self.linear_u_one(input_a))
        u_a2 = F.relu(self.linear_u_one2(u_a))
        u_b = F.relu(self.linear_u_two(input_b))
        u_b2 = F.relu(self.linear_u_two2(u_b))
        u_c = torch.add(u_a2, u_b2)
        weight = torch.exp(torch.tanh(u_c)).view(u_c.shape[0], -1)
        weight = (weight / torch.sum(weight, 1).view(-1, 1)).view(u_c.shape[0], -1, 1)
        weighted_input = torch.mul(input_a, weight)
        a = torch.sum(weighted_input, 1)
        b = input_b / (self.l - 1)
        b = b.view(b.shape[0], -1)
        if metric == 'dot':
            scores = metric_dot(a, b)
        elif metric == 'cosine':
            scores = metric_cosine(a, b)
        elif metric == 'euclid':
            scores = metric_euclid(a, b)
        elif metric == 'norm_euclid':
            scores = metric_norm_euclid(a, b)
        elif metric == 'manhattan':
            scores = metric_manhattan(a, b)
        elif metric == 'multi':
            scores = self.multi[0] * metric_euclid(a, b) + \
                     self.multi[1] * metric_dot(a, b) + \
                     self.multi[2] * metric_cosine(a, b)
        else:
            scores = metric_dot(a, b)
        return scores


class SchemasSGNN(BaseSGNN):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, limit, hidden_size):
        super(SchemasSGNN, self).__init__(limit, hidden_size)


class SkeletonSGNN(BaseSGNN):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, limit, hidden_size):
        super(SkeletonSGNN, self).__init__(limit, hidden_size)


def metric_dot(v0, v1):
    return torch.sum(v0 * v1, 1).view(-1, 1)


def metric_cosine(v0, v1):
    return F.cosine_similarity(v0, v1).view(-1, 1)


def metric_euclid(v0, v1):
    return -torch.norm(v0 - v1, 2, 1).view(-1, 1)


def metric_norm_euclid(v0, v1):
    v0 = v0 / torch.norm(v0, 2, 1).view(-1, 1)
    v1 = v1 / torch.norm(v1, 2, 1).view(-1, 1)
    return -torch.norm(v0 - v1, 2, 1).view(-1, 1)


def metric_manhattan(v0, v1):
    return -torch.sum(torch.abs(v0 - v1), 1).view(-1, 1)




class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adjacency_matrix, node_features):
        # 使用邻接矩阵进行传播
        x = torch.matmul(adjacency_matrix, node_features)
        x = self.linear(x)
        x = F.relu(x)  # 激活函数可以根据任务调整

        return x


class GNN(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.2):
        super(GNN, self).__init__()
        self.hidden_size = hidden_size
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_ah = Parameter(torch.Tensor(self.hidden_size))

        self.dropout = nn.Dropout(dropout_p)
        self.reset_parameters()

    def GNNCell(self, A, hidden, w_ih, w_hh, b_ih, b_hh, b_ah):
        input = torch.matmul(A.transpose(1, 2), hidden)
        input = self.dropout(input)
        gi = F.linear(input, w_ih, b_ih)
        gh = F.linear(hidden, w_hh, b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        hy = self.dropout(hy)
        return hy

    def forward(self, A, hidden):
        hidden1 = self.GNNCell(A, hidden, self.w_ih, self.w_hh, self.b_ih, self.b_hh, self.b_ah)
        hidden2 = self.GNNCell(A, hidden1, self.w_ih, self.w_hh, self.b_ih, self.b_hh, self.b_ah)
        return hidden2

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
