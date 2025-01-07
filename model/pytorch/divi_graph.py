import torch
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import networkx as nx

def compute_pcc_matrix(ts):
    num_nodes = ts.shape[0]
    epsilon = 1e-10  # 添加一个很小的常数以避免除零错误
    
    # 标准化时间序列数据
    ts_mean = torch.mean(ts, dim=1, keepdim=True)
    ts_std = torch.std(ts, dim=1, keepdim=True) + epsilon  # 添加epsilon避免除零错误
    ts_norm = (ts - ts_mean) / ts_std
    
    # 计算PCC矩阵
    pcc_matrix = torch.matmul(ts_norm, ts_norm.transpose(0, 1)) / ts.shape[1]
    
    # 处理自相关和数值稳定性问题
    pcc_matrix.fill_diagonal_(1.0)
    pcc_matrix[torch.isnan(pcc_matrix)] = 0
    pcc_matrix[torch.isinf(pcc_matrix)] = 0
    
    return pcc_matrix

def normalize_pcc_matrix(pcc_matrix):
    normalized_pcc_matrix = (pcc_matrix + 1) / 2
    return normalized_pcc_matrix

def split_graph_nor(adj_matrix, num_subgrids):
    num_nodes = adj_matrix.shape[0]
    nodes_per_subgrid = num_nodes // num_subgrids
    extra_nodes = num_nodes % num_subgrids
    
    split_indices = []
    for i in range(num_subgrids):
        start_idx = i * nodes_per_subgrid
        end_idx = (i + 1) * nodes_per_subgrid
        split_indices.append(torch.arange(start_idx, end_idx))
    
    for i in range(extra_nodes):
        split_indices[i] = torch.cat([split_indices[i], torch.tensor([num_nodes - extra_nodes + i])])
    
    split_indices = [indices.tolist() for indices in split_indices]

    # 保留完整的边信息，包括跨子图的边
    subgraphs = []
    for indices in split_indices:
        subgraph_matrix = adj_matrix.clone()  # 保留完整的邻接矩阵
        # 仅在该子图的行或列不相关的部分保留边
        mask = torch.ones_like(adj_matrix)
        mask[indices, :] = 0  # 行置零
        mask[:, indices] = 0  # 列置零
        subgraph_matrix[mask.bool()] = 0  # 删除不属于该子图的边
        subgraphs.append(subgraph_matrix)
    
    # 将split_indices列表转换为张量
    split_indices = torch.tensor([torch.tensor(indices) for indices in split_indices])
    
    return subgraphs, split_indices

def split_graph_nor(adj_matrix, num_subgrids):
    num_nodes = adj_matrix.shape[0]
    nodes_per_subgrid = num_nodes // num_subgrids
    extra_nodes = num_nodes % num_subgrids
    
    # 创建子图索引
    split_indices = []
    for i in range(num_subgrids):
        start_idx = i * nodes_per_subgrid
        end_idx = (i + 1) * nodes_per_subgrid
        split_indices.append(torch.arange(start_idx, end_idx))
    
    # 分配额外的节点
    for i in range(extra_nodes):
        split_indices[i] = torch.cat([split_indices[i], torch.tensor([num_nodes - extra_nodes + i])])
    
    # 保留子图对应的节点和边
    subgraphs = []
    for indices in split_indices:
        subgraph_matrix = adj_matrix[indices][:, indices].clone()  # 只保留该子图的节点和边
        subgraphs.append(subgraph_matrix)
    
    # 将split_indices列表转换为张量
    split_indices = torch.stack([torch.tensor(indices) for indices in split_indices])
    
    return subgraphs, split_indices

def split_graph_pcc(adj_matrix, ts, num_subgrids):
    num_nodes = adj_matrix.shape[0]
    if num_subgrids >= num_nodes:
        num_subgrids = num_nodes - 1
    
    # 计算PCC矩阵
    ts_combined = ts.mean(dim=0).t()
    pcc_matrix = compute_pcc_matrix(ts_combined)
    pcc_matrix = normalize_pcc_matrix(pcc_matrix)

    # 计算每个节点与其他节点的平均相关性
    avg_pcc = torch.mean(pcc_matrix, dim=1)
    
    # 根据平均相关性对节点排序
    sorted_indices = torch.argsort(avg_pcc, descending=True)
    
    # 初始化子图索引
    split_indices = [[] for _ in range(num_subgrids)]
    
    # 将节点均匀分配到子图中
    for i, node in enumerate(sorted_indices):
        split_indices[i % num_subgrids].append(node)
    
    # 转换为张量
    split_indices = [torch.tensor(indices) for indices in split_indices]

    # 生成子图，仅保留当前子图的节点和边
    subgraphs = []
    for indices in split_indices:
        subgraph_matrix = adj_matrix[indices][:, indices].clone()  # 子图中只保留相关节点和边
        subgraphs.append(subgraph_matrix)
    
    split_indices = torch.stack(split_indices)
    
    return subgraphs, split_indices

def split_graph_pcc_plus(adj_matrix, ts, num_subgrids, alpha=0.5):
    num_nodes = adj_matrix.shape[0]
    if num_subgrids >= num_nodes:
        num_subgrids = num_nodes - 1
    
    # 计算PCC矩阵
    ts_combined = ts.mean(dim=0).t()
    pcc_matrix = compute_pcc_matrix(ts_combined)
    pcc_matrix = normalize_pcc_matrix(pcc_matrix)
    
    # 计算边权重矩阵并进行标准化
    edge_weights = adj_matrix.clone()
    max_weight = torch.max(edge_weights)
    if max_weight > 0:
        edge_weights = edge_weights / max_weight
    
    # 组合PCC矩阵和边权重矩阵
    combined_matrix = alpha * pcc_matrix + (1 - alpha) * edge_weights

    # 计算每个节点与其他节点的平均值
    avg_combined = torch.mean(combined_matrix, dim=1)
    
    # 根据平均值对节点排序
    sorted_indices = torch.argsort(avg_combined, descending=True)
    
    # 初始化子图索引
    split_indices = [[] for _ in range(num_subgrids)]
    
    # 将节点均匀分配到子图中
    for i, node in enumerate(sorted_indices):
        split_indices[i % num_subgrids].append(node)

    # 转换为张量
    split_indices = [torch.tensor(indices) for indices in split_indices]

    # 生成子图，仅保留当前子图的节点和边
    subgraphs = []
    for indices in split_indices:
        subgraph_matrix = adj_matrix[indices][:, indices].clone()  # 子图中只保留相关节点和边
        subgraphs.append(subgraph_matrix)
    
    split_indices = torch.stack(split_indices)
    
    return subgraphs, split_indices

def split_graph_overlap_pcc(adj_matrix, ts, num_subgrids, overlap_percentage=0.5):
    num_nodes = adj_matrix.shape[0]

    # 计算PCC矩阵
    ts_combined = ts.mean(dim=0).t()
    pcc_matrix = compute_pcc_matrix(ts_combined)
    pcc_matrix = normalize_pcc_matrix(pcc_matrix)
    pcc_matrix.fill_diagonal_(0.0)  # 将对角线（自环）值设为0
    pcc_matrix[torch.isnan(pcc_matrix)] = 0
    pcc_matrix[torch.isinf(pcc_matrix)] = 0

    # 计算每个节点与其他节点的平均PCC值
    avg_pcc = torch.mean(pcc_matrix, dim=1)
    
    # 根据平均PCC值对节点排序
    sorted_indices = torch.argsort(avg_pcc, descending=True)
    
    # 初始化子图索引
    split_indices = [[] for _ in range(num_subgrids)]
    
    # 将节点均匀分配到子图中，优先将高PCC的节点分配到各子图
    for i, node in enumerate(sorted_indices):
        split_indices[i % num_subgrids].append(node)
    
    # 转换为张量
    split_indices = [torch.tensor(indices) for indices in split_indices]

    # 生成子图，仅保留当前子图的节点和边
    subgraphs = []
    for indices in split_indices:
        subgraph_matrix = adj_matrix[indices][:, indices].clone()
        subgraphs.append(subgraph_matrix)

    # 处理重叠部分，确保尽量不丢失重要边
    overlap_count = max(1, int(max(len(indices) for indices in split_indices) * overlap_percentage))

    for i in range(num_subgrids):
        subgrid_set = set(split_indices[i].tolist())
        
        # 按照PCC值，选择与当前子图有最大相关性的额外节点，进行重叠
        additional_nodes = []
        for node in sorted_indices:
            if node.item() not in subgrid_set:
                # 计算该节点与当前子图中节点的最大PCC值，选择高相关性的节点作为重叠节点
                node_pcc = pcc_matrix[node][split_indices[i]].max().item()
                if node_pcc > 0.5:  # 选择高于阈值的相关性边（这个阈值可以根据需要调整）
                    additional_nodes.append(node)
            if len(additional_nodes) == overlap_count:
                break

        # 将这些额外节点加入当前子图的索引
        split_indices[i] = torch.cat([split_indices[i], torch.tensor(additional_nodes)])
        subgrid_set.update(additional_nodes)

    # 生成重叠节点后的子图，仅保留相关节点和边
    subgraphs = []
    for indices in split_indices:
        subgraph_matrix = adj_matrix[indices][:, indices].clone()
        subgraphs.append(subgraph_matrix)

    split_indices = torch.stack(split_indices)

    return subgraphs, split_indices

def split_graph_overlap_pcc_plus(adj_matrix, ts, num_subgrids, overlap_percentage=0.5, alpha=0.5):
    num_nodes = adj_matrix.shape[0]

    # 计算PCC矩阵
    ts_combined = ts.mean(dim=0).t()
    pcc_matrix = compute_pcc_matrix(ts_combined)
    pcc_matrix = normalize_pcc_matrix(pcc_matrix)
    pcc_matrix.fill_diagonal_(0.0)  # 将对角线（自环）值设为0
    pcc_matrix[torch.isnan(pcc_matrix)] = 0
    pcc_matrix[torch.isinf(pcc_matrix)] = 0

    # 计算边权重矩阵并进行标准化
    edge_weights = adj_matrix.clone()
    max_weight = torch.max(edge_weights)
    if max_weight > 0:
        edge_weights = edge_weights / max_weight
    
    # 组合PCC矩阵和边权重矩阵
    combined_matrix = alpha * pcc_matrix + (1 - alpha) * edge_weights

    # 计算每个节点与其他节点的平均值
    avg_combined = torch.mean(combined_matrix, dim=1)
    
    # 根据平均值对节点排序
    sorted_indices = torch.argsort(avg_combined, descending=True)
    
    # 初始化子图索引
    split_indices = [[] for _ in range(num_subgrids)]
    
    # 将节点均匀分配到子图中，优先将高相关性或高权重边的节点分配到各子图
    for i, node in enumerate(sorted_indices):
        split_indices[i % num_subgrids].append(node)

    # 转换为张量
    split_indices = [torch.tensor(indices) for indices in split_indices]

    # 生成子图，仅保留当前子图的节点和边
    subgraphs = []
    for indices in split_indices:
        subgraph_matrix = adj_matrix[indices][:, indices].clone()
        subgraphs.append(subgraph_matrix)

    # 处理重叠部分
    overlap_count = max(1, int(max(len(indices) for indices in split_indices) * overlap_percentage))

    for i in range(num_subgrids):
        subgrid_set = set(split_indices[i].tolist())
        
        # 按照综合PCC和权重值，选择与当前子图有最大相关性的节点
        additional_nodes = []
        for node in sorted_indices:
            if node.item() not in subgrid_set:
                # 计算该节点与当前子图中节点的最大组合相关性
                node_combined = combined_matrix[node][split_indices[i]].max().item()
                if node_combined > 0.5:  # 选择高于阈值的边作为重叠节点
                    additional_nodes.append(node)
            if len(additional_nodes) == overlap_count:
                break

        # 将这些额外节点加入当前子图的索引
        split_indices[i] = torch.cat([split_indices[i], torch.tensor(additional_nodes)])
        subgrid_set.update(additional_nodes)

    # 生成重叠节点后的子图，仅保留相关节点和边
    subgraphs = []
    for indices in split_indices:
        subgraph_matrix = adj_matrix[indices][:, indices].clone()
        subgraphs.append(subgraph_matrix)

    split_indices = torch.stack(split_indices)

    return subgraphs, split_indices


def split_time_series(ts, split_indices):
    # 假设split_indices已经是一个张量
    ts_sublists = torch.stack([ts[:, :, indices] for indices in split_indices], dim=0)
    
    return ts_sublists

def merge_time_series(ts_sublists, split_indices, num_nodes, num_batches, seq_length):
    device = ts_sublists.device
    
    # 初始化合并后的时间序列张量
    merged_ts = torch.zeros((num_batches, seq_length, num_nodes), device=device)
    
    # 使用高级索引合并子图时间序列
    for i, indices in enumerate(split_indices):
        merged_ts[:, :, indices] = ts_sublists[i]
    
    return merged_ts

def visualize_graph(adj_matrix, title, save_path):
    g = nx.from_numpy_array(adj_matrix.numpy())
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(g)
    nx.draw(g, pos, with_labels=True, node_color="skyblue", node_size=500, edge_color="black", width=1.0)
    edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def visualize_time_series(ts1, ts2, title1, title2, save_path):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for i in range(ts1.shape[1]):
        plt.plot(ts1[:, i].numpy(), label=f'Node {i}')
    plt.title(title1)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i in range(ts2.shape[1]):
        plt.plot(ts2[:, i].numpy(), label=f'Node {i}')
    plt.title(title2)
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()

def visualize_subgraphs(subgraphs, title, save_path):
    fig, axes = plt.subplots(1, len(subgraphs), figsize=(15, 5))
    for i, ax in enumerate(axes):
        subgraph_matrix = subgraphs[i]
        g = nx.from_numpy_array(subgraph_matrix.numpy())
        pos = nx.spring_layout(g)
        nx.draw(g, pos, with_labels=True, node_color="skyblue", node_size=500, edge_color="black", width=1.0, ax=ax)
        ax.set_title(f"Subgraph {i}")
    plt.suptitle(title)
    plt.savefig(save_path)
    plt.close()
    
def print_subgraph_node_counts(split_indices, method_name):
    print(f"{method_name} 方法生成的子网节点数量:")
    for i, indices in enumerate(split_indices):
        print(f"子网 {i} 节点数量: {len(indices)}")

if __name__ == "__main__":
    
    num_nodes = 10
    adjacency_matrix = torch.rand((num_nodes, num_nodes), dtype=torch.float32)
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.t()) / 2  # 保证对称性
    adjacency_matrix.fill_diagonal_(0)  # 对角线置零
    
    ts_length = 12
    num_batches = 64
    ts = torch.rand((num_batches, ts_length, num_nodes))
    
    num_subgrids = 2
    overlap_percentage = 0.0  # 传入overlap参数

    # 测试不同的划分函数
    subgraphs_nor, split_indices_nor = split_graph_nor(adjacency_matrix, num_subgrids=num_subgrids)
    ts_sublists_nor = split_time_series(ts, split_indices_nor)
    merged_ts_nor = merge_time_series(ts_sublists_nor, split_indices_nor, num_nodes, num_batches, ts_length)
    
    subgraphs_pcc, split_indices_pcc = split_graph_pcc(adjacency_matrix, ts, num_subgrids=num_subgrids)
    ts_sublists_pcc = split_time_series(ts, split_indices_pcc)
    merged_ts_pcc = merge_time_series(ts_sublists_pcc, split_indices_pcc, num_nodes, num_batches, ts_length)
    
    subgraphs_pcc_plus, split_indices_pcc_plus = split_graph_pcc_plus(adjacency_matrix, ts, num_subgrids=num_subgrids, alpha=0.5)
    ts_sublists_pcc_plus = split_time_series(ts, split_indices_pcc_plus)
    merged_ts_pcc_plus = merge_time_series(ts_sublists_pcc_plus, split_indices_pcc_plus, num_nodes, num_batches, ts_length)
    
    subgraphs_overlap_pcc, split_indices_overlap_pcc = split_graph_overlap_pcc(adjacency_matrix, ts, num_subgrids=num_subgrids, overlap_percentage=overlap_percentage)
    ts_sublists_overlap_pcc = split_time_series(ts, split_indices_overlap_pcc)
    merged_ts_overlap_pcc = merge_time_series(ts_sublists_overlap_pcc, split_indices_overlap_pcc, num_nodes, num_batches, ts_length)
    
    subgraphs_overlap_pcc_plus, split_indices_overlap_pcc_plus = split_graph_overlap_pcc_plus(adjacency_matrix, ts, num_subgrids=num_subgrids, alpha=0.5, overlap_percentage=overlap_percentage)
    ts_sublists_overlap_pcc_plus = split_time_series(ts, split_indices_overlap_pcc_plus)
    merged_ts_overlap_pcc_plus = merge_time_series(ts_sublists_overlap_pcc_plus, split_indices_overlap_pcc_plus, num_nodes, num_batches, ts_length)
    
    print("原图节点数:", num_nodes)
    print_subgraph_node_counts(split_indices_nor, "NOR")
    print_subgraph_node_counts(split_indices_pcc, "PCC")
    print_subgraph_node_counts(split_indices_pcc_plus, "PCC+")
    print_subgraph_node_counts(split_indices_overlap_pcc, "Overlap PCC")
    print_subgraph_node_counts(split_indices_overlap_pcc_plus, "Overlap PCC+")
    
    print("NOR 合并后的时间序列形状:", merged_ts_nor.shape)
    print("PCC 合并后的时间序列形状:", merged_ts_pcc.shape)
    print("PCC+ 合并后的时间序列形状:", merged_ts_pcc_plus.shape)
    print("Overlap PCC 合并后的时间序列形状:", merged_ts_overlap_pcc.shape)
    print("Overlap PCC+ 合并后的时间序列形状:", merged_ts_overlap_pcc_plus.shape)
    
    if torch.equal(ts, merged_ts_nor):
        print("NOR 合并正确")
    else:
        print("NOR 合并错误")
    
    if torch.equal(ts, merged_ts_pcc):
        print("PCC 合并正确")
    else:
        print("PCC 合并错误")
    
    if torch.equal(ts, merged_ts_pcc_plus):
        print("PCC+ 合并正确")
    else:
        print("PCC+ 合并错误")

    if torch.equal(ts, merged_ts_overlap_pcc):
        print("Overlap PCC 合并正确")
    else:
        print("Overlap PCC 合并错误")
    
    if torch.equal(ts, merged_ts_overlap_pcc_plus):
        print("Overlap PCC+ 合并正确")
    else:
        print("Overlap PCC+ 合并错误")
    
    visualize_time_series(ts[0, :, :num_nodes], merged_ts_nor[0, :, :num_nodes], 
                          "Original Time Series", "Merged Time Series (NOR)", "view/time_series_comparison_nor.png")
    visualize_time_series(ts[0, :, :num_nodes], merged_ts_pcc[0, :, :num_nodes], 
                          "Original Time Series", "Merged Time Series (PCC)", "view/time_series_comparison_pcc.png")
    visualize_time_series(ts[0, :, :num_nodes], merged_ts_pcc_plus[0, :, :num_nodes], 
                          "Original Time Series", "Merged Time Series (PCC+)", "view/time_series_comparison_pcc_plus.png")
    visualize_time_series(ts[0, :, :num_nodes], merged_ts_overlap_pcc[0, :, :num_nodes], 
                          "Original Time Series", "Merged Time Series (Overlap PCC)", "view/time_series_comparison_overlap_pcc.png")
    visualize_time_series(ts[0, :, :num_nodes], merged_ts_overlap_pcc_plus[0, :, :num_nodes], 
                          "Original Time Series", "Merged Time Series (Overlap PCC+)", "view/time_series_comparison_overlap_pcc_plus.png")
    
    visualize_graph(adjacency_matrix, "Original Graph", "view/original_graph.png")
    visualize_subgraphs(subgraphs_nor, "Subgraphs (NOR)", "view/subgraphs_nor.png")
    visualize_subgraphs(subgraphs_pcc, "Subgraphs (PCC)", "view/subgraphs_pcc.png")
    visualize_subgraphs(subgraphs_pcc_plus, "Subgraphs (PCC+)", "view/subgraphs_pcc_plus.png")
    visualize_subgraphs(subgraphs_overlap_pcc, "Subgraphs (Overlap PCC)", "view/subgraphs_overlap_pcc.png")
    visualize_subgraphs(subgraphs_overlap_pcc_plus, "Subgraphs (Overlap PCC+)", "view/subgraphs_overlap_pcc_plus.png")
