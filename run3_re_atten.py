import pickle
import time
import torch
import os
from utils.util import add_graph, convert_graph, my_sub_graph, random_sort
from models.future import Future
from models.sgnn3_re_atten import SGNN
import networkx as nx
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import torch.nn.functional as F
import json
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
# from thop import profile
train_list = []
test_list = []
writer = SummaryWriter(log_dir="./logs")
now_time = str(time.strftime("%Y-year-%m-month-%d-day-%H-hour-%M-minute-%S-second"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 训练集准备
# with open('data/new/test/test3.json', 'r') as file:
#     test_data = json.load(file)
# with open('data/new/train/train3.json', 'r') as file:
#     train_data = json.load(file)
# with open('data/new/train/traingraph11_4.json', 'r') as file:
#     datagraph = json.load(file)
# traingraph = datagraph[:5700]
with open('data/new/train/traingraph15_5.json', 'r') as file:
    datagraph = json.load(file)
traingraph = datagraph[:6200]
# traingraph = datagraph[:2]
# test_data = test_data[:20]
# ontology = read_ontology_excel("./data/kairos-ontology.xlsx")
# future = Future(ontology)
# future.to(future.device)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
# 数据集分割：
# 初始化 11x11 的零矩阵

graphlen = 15
node_num = 14
# 填充邻接矩阵
for i, graph in enumerate(traingraph):
    adj_matrix = np.zeros((graphlen, graphlen), dtype=int)
    for edge in graph[1]:
        node1, node2 = edge
        adj_matrix[node1 - 1][node2 - 1] = 1
        adj_matrix[node2 - 1][node1 - 1] = 1  # 无向图对称填充
    adj_matrix = adj_matrix[:node_num, :node_num]
    traingraph[i][1] = adj_matrix
    traingraph[i].append(traingraph[i][0][node_num])
    traingraph[i][0] = traingraph[i][0][:node_num]

# with open('./data/new/{}/{}{}.json'.format('train', 'train', 'graph11_6'), 'w') as file:
#     json.dump(traingraph, file)
# 数据分类
num_categories = 10
category_size = len(traingraph) // num_categories  # 每个类别 500 个样本
categories = {i: [] for i in range(num_categories)}
# 按类别划分数据
for idx, sample in enumerate(traingraph):
    category_index = idx // category_size  # 计算样本属于哪个类别
    categories[category_index].append(sample)
# 初始化数据集
train_data, test_data, val_data = [], [], []
# 对每个类别进行分层抽样
np.random.seed(42)
for category, items in categories.items():
    np.random.shuffle(items)  # 打乱顺序

    train_size = int(0.8 * len(items))
    test_size = int(0.1 * len(items))

    train_data.extend(items[:train_size])
    test_data.extend(items[train_size:train_size + test_size])
    val_data.extend(items[train_size + test_size:])
# view_data = traingraph[6]
sgnn = SGNN(sch_hidden_size=node_num, ske_hidden_size=16).to(device)
nparams = sum([p.nelement() for p in sgnn.parameters()])
optimizer = torch.optim.Adam(sgnn.parameters(), lr=3e-4, weight_decay=0)


batch_size = 1
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
iter_wrapper_train = (lambda x: tqdm(x, total=len(train_loader), ncols=80))
iter_wrapper_test = (lambda x: tqdm(x, total=len(test_loader), ncols=80))
step = 0
train = 0
test = 1





def ewc_loss(sgnn, fisher, prev_params, lambda_ewc):
    loss = 0.0
    for name, param in sgnn.named_parameters():
        if "linear1" in name:  # 只对 linear1 计算 EWC
            loss += (fisher[name] * (param - prev_params[name]) ** 2).sum()
    return lambda_ewc * loss

for epoch in range(300):
    if train == 1:
        # 训练
        # sgnn.load_state_dict(torch.load('models/output/example_model{}.pth'.format(109)))
        sgnn.train()
        sgnn.zero_grad()
        optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        entry_loss_all = 0
        current_loss = 0
        current_acc = 0
        score_list_sub_adj_all = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        score_list_adj_all = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        mrr_all, hit_k1_all, hit_k3_all, hit_k10_all = 0,0,0,0
        # 初始化计算费舍尔矩阵
        # fisher = {}
        # prev_params = {}
        # # 初始化 Fisher 矩阵
        # for name, param in sgnn.named_parameters():
        #     fisher[name] = torch.zeros_like(param)
        #     prev_params[name] = param.clone().detach()

        # for i, li in tqdm(enumerate(train_data)):
        for s, (enity, adj, label) in iter_wrapper_train(enumerate(train_loader)):


            Results = sgnn(enity, adj, label, s)
            step_loss_all, accuracy_gen, score_list_sub_adj, score_list_adj,  mrr, hit_k1, hit_k3, hit_k10 = Results
            # loss1 = sgnn.loss_node(score_i, target)
            # loss2 = sgnn.loss_path(score_i, path_target)



            # Fisher 近似值 = 梯度平方
            # for name, param in sgnn.named_parameters():
            #     fisher[name] += param.grad ** 2  # 近似 Fisher 矩阵
            #
            # reg_loss = ewc_loss(sgnn, fisher, prev_params, lambda_ewc)

            loss = step_loss_all
            sgnn.zero_grad()
            # ith torch.autograd.detect_anomaly():
                # loss.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            optimizer.step()
            current_loss += loss.item()
            current_acc += accuracy_gen.item()
            mrr_all += mrr
            hit_k1_all += hit_k1
            hit_k3_all += hit_k3
            hit_k10_all += hit_k10

            score_list_sub_adj_all = list(map(lambda x, y: x + y, score_list_sub_adj, score_list_sub_adj_all))
            score_list_adj_all = list(map(lambda x, y: x + y, score_list_adj, score_list_adj_all))
            # score_list_sub_adj_all += score_list_sub_adj
            # score_list_adj_all += score_list_adj
            # accu1 = sgnn.accu_node(score_i, target)
            # accu2 = sgnn.accu_path(score_i, path_target)
            # print("loss1:", float(loss1), "loss2:", float(loss2), "accu1:", float(accu1), "accu2:", float(accu2))
            # print("loss:", float(loss), "accu:", float(accuracy_gen), )
            writer.add_scalars("{}/train/loss".format(now_time), {"node_loss": loss}, step)
            writer.add_scalars("{}/train/loss".format(now_time), {"node_accuracy": accuracy_gen}, step)
            # writer.add_scalars("{}/train/loss".format(now_time), {"graph_loss": graph_loss}, step)
            # writer.add_scalars("{}/train/loss".format(now_time), {"node_choose_loss": node_choose_loss}, step)
            step += 1
            # writer.add_scalars("{}/train/accu".format(now_time), {"node": accu1}, step)
            # writer.add_scalars("{}/train/accu".format(now_time), {"path": accu2}, step)
        current_loss_avg = current_loss / s
        current_acc_avg = current_acc / s

        mrr_avg = mrr_all / s
        hit_k1_avg = hit_k1_all / s
        hit_k3_all = hit_k3_all / s
        hit_k10_avg = hit_k10_all / s


        score_list_sub_adj_all = np.array(score_list_sub_adj_all)
        score_list_adj_all = np.array(score_list_adj_all)
        s_np = np.array([s])
        score_list_sub_adj_avg = score_list_sub_adj_all / s_np
        score_list_adj_avg = score_list_adj_all / s_np
        # entry_loss_avg = entry_loss_all / s
        if epoch >= 2 and epoch % 1 == 0:
            torch.save(sgnn.state_dict(), 'models/output3_14_re_atten/example_model{}.pth'.format(epoch))
        print("epoch:", epoch, "loss:", current_loss_avg, "acc:", current_acc_avg,
              "mrr_avg:", mrr_avg, "hit_k1_avg:", hit_k1_avg, "hit_k3_all:", hit_k3_all, "hit_k10_avg:", hit_k10_avg,
              "score_sub_adj:", score_list_sub_adj_avg, "score_adj:", score_list_adj_avg)
        writer.add_scalars("{}/train/loss".format(now_time), {"current_loss_avg": current_loss_avg}, epoch)
        writer.add_scalars("{}/train/loss".format(now_time), {"current_acc_avg": current_acc_avg}, epoch)
        # writer.add_scalars("{}/train/loss".format(now_time), {"entry_loss_avg": entry_loss_avg}, epoch)
    if test == 1:
        # 测试
        # 200：87.29; 240:89.3

        sgnn.load_state_dict(torch.load('models/output3_14_re_atten/example_model{}.pth'.format(4)))
        sgnn.eval()
        entry_loss_all_test = 0
        current_loss_test = 0
        current_acc_test = 0
        structure_similarity_score_test = 0
        attribute_similarity_score_test = 0
        similarity_score_test = 0
        edge_diff_test = 0
        mrr_all, hit_k1_all, hit_k3_all, hit_k10_all = 0, 0, 0, 0
        score_list_sub_adj_all = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        score_list_adj_all = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # for i, li in tqdm(enumerate(train_data)):
        for s, (enity, adj, label) in iter_wrapper_test(enumerate(test_loader)):

            Results = sgnn(enity, adj, label, s)
            step_loss_all, accuracy_gen,  score_list_sub_adj, score_list_adj,  mrr, hit_k1, hit_k3, hit_k10 = Results
            # loss1 = sgnn.loss_node(score_i, target)
            # loss2 = sgnn.loss_path(score_i, path_target)
            loss = step_loss_all
            current_loss_test += loss.item()
            current_acc_test += accuracy_gen.item()

            mrr_all += mrr
            hit_k1_all += hit_k1
            hit_k3_all += hit_k3
            hit_k10_all += hit_k10

            score_list_sub_adj_all = list(map(lambda x, y: x + y, score_list_sub_adj, score_list_sub_adj_all))
            score_list_adj_all = list(map(lambda x, y: x + y, score_list_adj, score_list_adj_all))
            # entry_loss_all_test += node_choose_loss.item()
            # attribute_similarity_score_test += attribute_similarity_score.item()
            # structure_similarity_score_test += structure_similarity_score.item()
            # similarity_score_test += similarity_score.item()
            # edge_diff_test += edge_diff.item()
            # accu1 = sgnn.accu_node(score_i, target)
            # accu2 = sgnn.accu_path(score_i, path_target)
            # print("loss1:", float(loss1), "loss2:", float(loss2), "accu1:", float(accu1), "accu2:", float(accu2))
            # print("loss:", float(loss), "accu:", float(accuracy_gen), )
            writer.add_scalars("{}/test/loss".format(now_time), {"node_loss": loss}, step)
            writer.add_scalars("{}/test/loss".format(now_time), {"node_accuracy": accuracy_gen}, step)
            # writer.add_scalars("{}/test/loss".format(now_time), {"graph_loss": graph_loss}, step)
            # writer.add_scalars("{}/test/loss".format(now_time), {"node_choose_loss": node_choose_loss}, step)
            step += 1
            # writer.add_scalars("{}/train/accu".format(now_time), {"node": accu1}, step)
            # writer.add_scalars("{}/train/accu".format(now_time), {"path": accu2}, step)
        current_loss_avg = current_loss_test / (s+1)
        current_acc_avg = current_acc_test / (s+1)
        entry_loss_avg = entry_loss_all_test / (s+1)
        attribute_similarity_score_avg = attribute_similarity_score_test / (s+1)
        structure_similarity_score_avg = structure_similarity_score_test / (s+1)
        similarity_score_avg = similarity_score_test / (s+1)
        edge_diff_avg = edge_diff_test / (s+1)

        mrr_avg = mrr_all / (s+1)
        hit_k1_avg = hit_k1_all / (s+1)
        hit_k3_all = hit_k3_all / (s+1)
        hit_k10_avg = hit_k10_all / (s+1)

        score_list_sub_adj_all = np.array(score_list_sub_adj_all)
        score_list_adj_all = np.array(score_list_adj_all)
        s_np = np.array([(s+1)])
        score_list_sub_adj_avg = score_list_sub_adj_all / s_np
        score_list_adj_avg = score_list_adj_all / s_np
        print("s:", s)
        print("epoch:", epoch, "loss:", current_loss_avg, "test_acc:", current_acc_avg,
              "mrr_avg:", mrr_avg, "hit_k1_avg:", hit_k1_avg, "hit_k3_all:", hit_k3_all, "hit_k10_avg:", hit_k10_avg,
              "score_sub_adj:", score_list_sub_adj_avg, "score_adj:", score_list_adj_avg)
        writer.add_scalars("{}/test/loss".format(now_time), {"current_loss_avg": current_loss_avg}, epoch)
        writer.add_scalars("{}/test/loss".format(now_time), {"current_acc_avg": current_acc_avg}, epoch)
        writer.add_scalars("{}/test/loss".format(now_time), {"entry_loss_avg": entry_loss_avg}, epoch)
        writer.add_scalars("{}/test/loss".format(now_time), {"attribute_similarity_score_avg": attribute_similarity_score_avg}, epoch)
        writer.add_scalars("{}/test/loss".format(now_time), {"structure_similarity_score_avg": structure_similarity_score_avg}, epoch)
        writer.add_scalars("{}/test/loss".format(now_time), {"similarity_score_avg": similarity_score_avg}, epoch)
        writer.add_scalars("{}/test/loss".format(now_time), {"edge_diff": edge_diff_avg}, epoch)
