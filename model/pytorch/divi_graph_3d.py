import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import networkx as nx
import dgl
from matplotlib.colors import LinearSegmentedColormap
import scipy.sparse as sp
import math
import math
import community as community_louvain
import pandas as pd


def compute_pcc_matrix(ts):
    num_nodes = ts.shape[0]
    epsilon = 1e-10  
    
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

def split_time_series(ts, split_indices):

    ts_sublists = torch.stack([ts[:, :, indices] for indices in split_indices], dim=0)
    return ts_sublists

def merge_time_series(ts_sublists, split_indices, num_nodes, num_batches, seq_length):
    device = ts_sublists.device
    merged_ts = torch.zeros((num_batches, seq_length, num_nodes), device=device)

    for i, indices in enumerate(split_indices):
        merged_ts[:, :, indices] = ts_sublists[i]
    return merged_ts


def split_graph_3d(adj, G, ts, num_subgrids):
    np.set_printoptions(threshold=np.inf)

    num_nodes = G.number_of_nodes()
    max_nodes_per_subgraph = math.ceil(num_nodes / num_subgrids)  

    adj_matrix = adj

    adj_matrix_sparse = sp.coo_matrix(adj_matrix)
   
    nx_G = nx.from_numpy_array(adj_matrix)


    # Step 2: 计算连通组件

    partition = community_louvain.best_partition(nx_G)

    community_dict = {}
    for node, community_id in partition.items():
        if community_id not in community_dict:
            community_dict[community_id] = []
        community_dict[community_id].append(node)
    sorted_community_ids  = sorted(community_dict.keys())
  
    subgraph_nodes = [[] for _ in range(num_subgrids)]  # 创建空子图列表

    # Step 4: 将节点按社区 ID 顺序均衡分配到子图
    total_nodes = sum(len(nodes) for nodes in community_dict.values())  # 获取所有节点的总数
    nodes_per_subgraph = total_nodes // num_subgrids  # 每个子图的基本节点数
    remaining_nodes = total_nodes % num_subgrids  # 需要额外添加的节点数量

    # 将剩余的节点均匀分配到前面的子图中
    community_ids_with_extra = sorted_community_ids[:remaining_nodes]  # 选择要多分配的社区ID

    # Step 5: 按社区顺序分配节点
    sorted_community_ids = sorted(community_dict.keys())  # 对社区ID进行排序
    current_subgraph_index = 0  # 子图索引，初始从第一个子图开始

    for community_id in sorted_community_ids:
        nodes_in_community = community_dict[community_id]  # 当前社区的所有节点
        # 对社区中的节点进行分配
        for node in nodes_in_community:
            subgraph_nodes[current_subgraph_index].append(node)  # 将节点添加到对应子图
            # 判断是否需要调整子图分配的节点数量
            if len(subgraph_nodes[current_subgraph_index]) >= nodes_per_subgraph + (1 if current_subgraph_index in community_ids_with_extra else 0):
                current_subgraph_index += 1  # 当前子图节点数已达到目标，切换到下一个子图

    # Step 5: 补充节点，使每个子图的节点数量达到 max_nodes_per_subgraph
    for i, subgraph in enumerate(subgraph_nodes):
        current_size = len(subgraph)
        if current_size < max_nodes_per_subgraph:
            required_nodes = max_nodes_per_subgraph - current_size
            print(f"Subgraph {i} needs {required_nodes} more nodes")

            neighbors_to_add = []  # 用于存放需要补充的节点
            # 寻找相邻子图的邻居节点进行补充
            for j in range(num_subgrids):
                if len(neighbors_to_add) >= required_nodes:
                    break
                if i != j:  # 不从当前子图中补充
                    # 假设你有 `adj_matrix` 或其他方式来访问邻居节点
                    for node in subgraph_nodes[j]:
                        # 检查该节点是否是当前子图的邻居
                        if node not in subgraph and len(neighbors_to_add) < required_nodes:
                            neighbors_to_add.append(node)

            # 更新子图，补充邻居节点
            subgraph_nodes[i].extend(neighbors_to_add)

    
    # 创建 DGL 子图
    subgraphs = []
    for nodes in subgraph_nodes:
        # 根据节点列表从原图中创建子图
        subgraph = G.subgraph(nodes)
        subgraphs.append(subgraph)

        
    cross_edges = []
    # 遍历每对子图的节点对 (i, j)，检查是否有边，并确保它们不在同一子图中
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # 检查原图中是否有边
            if G.has_edges_between(i, j):
                # 检查边 (i, j) 是否跨子图
                if not any(i in split and j in split for split in subgraph_nodes):
                    cross_edges.append((i, j))  # 保留跨子图的边

    # 将跨子图的边转换为张量
    cross_edges = torch.tensor(cross_edges, dtype=torch.long)

    # Step 7: 返回结果
    return subgraphs, subgraph_nodes, cross_edges
    

def split_node_to_subgraph(node, subgraphs):
    """
    根据节点索引确定该节点属于哪个子图。
    
    参数:
    - node: (int) 节点索引
    - subgraphs: (list) 每个子图包含的节点集合
    
    返回:
    - subgraph_id: (int) 该节点所属的子图索引
    """
    for i, subgraph in enumerate(subgraphs):
        if node in subgraph:
            return i
    return -1  # 如果没有找到节点，返回 -1，表示节点未分配到任何子图


    
def extract_cross_edge_time_series(ts, cross_edges):
    cross_edge_ts_list = []
    for edge in cross_edges:
        u, v = edge  # 边的两端节点
        u_ts = ts[:, :, u]
        v_ts = ts[:, :, v]
        cross_edge_ts_list.append((u_ts, v_ts))
    return cross_edge_ts_list

def save_cross_edge_time_series(cross_edge_ts_list):
    torch_cross_edge_ts_list = []
    for u_ts, v_ts in cross_edge_ts_list:
        edge_ts = torch.stack((u_ts, v_ts), dim=-1)  # 形状为 (num_batches, ts_length, 2)
        torch_cross_edge_ts_list.append(edge_ts)
    torch_cross_edge_ts = torch.stack(torch_cross_edge_ts_list, dim=0)
    return torch_cross_edge_ts


def generate_sin_cos_time_series(num_batches, ts_length, num_nodes):
    """
    用不同频率的sin和cos函数生成时间序列数据。
    """
    time_series = torch.zeros((num_batches, ts_length, num_nodes), dtype=torch.float32)
    t = torch.linspace(0, 2 * np.pi, ts_length)  # 时间步长

    # 为每个节点生成不同频率和相位的sin/cos波形
    for node in range(num_nodes):
        freq = 0.5 + 0.1 * node  # 不同节点的频率变化
        phase = node * np.pi / num_nodes  # 不同节点的相位变化
        wave = torch.sin(freq * t + phase) + torch.cos(freq * t - phase)
        time_series[:, :, node] = wave.repeat(num_batches, 1)  # 复制到所有batch中
    
    return time_series



def visualize_subgraph_relation_graph(subgraph_relation_graph):
    """
    可视化子图关系图。
    
    参数:
    - subgraph_relation_graph: DGLGraph, 子图关系图。
    """
    # 将 DGL 图转换为 NetworkX 图
    nx_graph = dgl.to_networkx(subgraph_relation_graph, edge_attrs=['weight'])
    
    # 获取节点位置布局（如spring布局）
    pos = nx.spring_layout(nx_graph)  # spring_layout 使用弹簧算法进行布局
    
    # 绘制节点
    plt.figure(figsize=(10, 8))
    nx.draw(
        nx_graph, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=12
    )
    
    # 为每条边标注权重
    edge_labels = nx.get_edge_attributes(nx_graph, 'weight')
    nx.draw_networkx_edge_labels(
        nx_graph, pos, edge_labels=edge_labels, font_size=10
    )
    
    # 显示图
    save_path="view/subgraph_relation_graph.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def visualize_time_series(ts1, ts2, title1, title2, save_path):
    plt.figure(figsize=(12, 6))
    
    # 创建蓝到粉的渐变颜色映射
    cmap = LinearSegmentedColormap.from_list("pink_purple", ["#9795f0", "#fbc8d4"], N=ts1.shape[1])
    
    # 绘制第一个时间序列图
    plt.subplot(1, 2, 1)
    for i in range(ts1.shape[1]):
        plt.plot(ts1[:, i].numpy(), label=f'Node {i}', color=cmap(i))
    plt.title(title1)
    
    # 绘制第二个时间序列图
    plt.subplot(1, 2, 2)
    for i in range(ts2.shape[1]):
        plt.plot(ts2[:, i].numpy(), label=f'Node {i}', color=cmap(i))
    plt.title(title2)
    
    # 调整整体布局，只显示一个图例，并且将其放在图外，避免遮挡图
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize='small')
    
    # 保存图像
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def visualize_subgraphs_3d(subgraphs, cross_edges, num_subgrids, save_path="view/subgraph_3d.png", dpi=300):
    # 创建高分辨率的图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 设置整体布局比例
    ax.set_box_aspect([1, 1, 1])

    # 设置颜色映射
    color_map = plt.get_cmap("viridis")
    colors = color_map(np.linspace(0, 1, num_subgrids))
    node_positions = {}
    subgraph_indices = {}

    for i, subgraph in enumerate(subgraphs):
        indices = subgraph.ndata['_ID']  # 从 DGL 子图中获取原始节点索引
        num_nodes = len(indices)

        # 坐标生成
        x = np.random.uniform(i * 30, (i + 1) * 30, num_nodes)
        y = np.random.uniform(i * 100, (i + 1) * 100, num_nodes)
        z = np.random.uniform(i * 10, (i + 1) * 10, num_nodes)

        # 保存节点位置并绘制节点
        for idx, (xi, yi, zi) in zip(indices, zip(x, y, z)):
            node_positions[idx.item()] = (xi, yi, zi)
            subgraph_indices[idx.item()] = i  # 保存节点所属的子图索引
            ax.scatter(xi, yi, zi, color=colors[i], s=30, alpha=0.7, label=f'Subgraph {i+1}' if idx == indices[0] else "")

        # 绘制子图内的边
        edges = subgraph.edges()
        for u, v in zip(edges[0], edges[1]):
            u_idx = indices[u].item()
            v_idx = indices[v].item()
            pos_u = node_positions[u_idx]
            pos_v = node_positions[v_idx]
            ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
                    color=colors[i], alpha=0.6, linewidth=3)

    # 绘制跨子图的边
    for u, v in cross_edges:
        pos_u = node_positions[u.item()]
        pos_v = node_positions[v.item()]
        ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
                color='gray', linestyle='--', alpha=0.4, linewidth=1)

    # 设置视角
    ax.view_init(elev=15, azim=0)

    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')

    # 确保保存目录存在
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()

    # 返回子图颜色映射和节点所属子图索引
    return color_map, subgraph_indices


def visualize_original_graph(adj_matrix, subgraph_indices, color_map, num_subgrids, location_file=None, save_path="view/original_graph.png"):
    """
    可视化原始图（节点和边），节点位置根据地理经纬度，节点和边颜色与子图颜色一致。

    参数:
    - adj_matrix: 原始图的邻接矩阵
    - subgraph_indices: 每个节点对应的子图索引
    - color_map: 子图的颜色映射
    - num_subgrids: 子图数量
    - location_file: 包含经纬度信息的CSV文件路径
    - save_path: 图片保存路径
    """
    # 从CSV文件加载节点位置
    #location_file="data/sensor_graph/graph_sensor_locations.csv"
    #location_file="data/sensor_graph/graph_sensor_locations_bay.csv"
    G = nx.from_numpy_array(adj_matrix)
    G.remove_edges_from(nx.selfloop_edges(G)) 

    if location_file and os.path.exists(location_file):
        locations = pd.read_csv(location_file)
        node_positions = {row["index"]: (row["longitude"], row["latitude"]) for _, row in locations.iterrows()}
    else:
        node_positions = nx.spring_layout(G, seed=42)  # 使用 spring 自动布局
    # 生成 networkx 图

       # 定义颜色映射
    colors = color_map(np.linspace(0, 1, num_subgrids))

    # 存储节点的最终颜色
    node_colors = []
    
    # 遍历每个节点，检查它是否在多个子图中
    for node in G.nodes():
        # 获取当前节点的子图索引
        node_subgraphs = subgraph_indices[node]

        if isinstance(node_subgraphs, list):  # 如果节点属于多个子图
            # 混合这些子图的颜色，取平均值
            mixed_color = np.mean([colors[i] for i in node_subgraphs], axis=0)
            node_colors.append(mixed_color)
        else:  # 如果节点只属于一个子图
            node_colors.append(colors[node_subgraphs])

    # 将节点颜色转换为 matplotlib 可用的格式
    node_colors = np.array(node_colors)

    # 为每条边分配颜色（根据起点和终点的平均颜色）
    edge_colors = []
    for u, v in G.edges():
        color_u = colors[subgraph_indices[u]]
        color_v = colors[subgraph_indices[v]]
        edge_color = (color_u + color_v) / 2  # 平均颜色
        edge_colors.append(edge_color)

    # 绘制图形
    plt.figure(figsize=(10, 6))
    nx.draw(
        G, pos=node_positions, with_labels=False, node_color=node_colors, node_size=30,
        edge_color=edge_colors, alpha=0.4, node_shape='o', width=1.0
    )
    nx.draw_networkx_nodes(
        G, pos=node_positions, node_color=node_colors, alpha=0.7, node_size=30
    )
    plt.title("Original Graph with Geographical Positions")

    # 确保保存目录存在
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    num_nodes = 20
    num_subgrids = 3
    adj_matrix = torch.rand((num_nodes, num_nodes), dtype=torch.float32)
    adj_matrix = (adj_matrix + adj_matrix.t()) / 2  # 保证对称性
    adj_matrix.fill_diagonal_(0)  # 对角线置零
    
    ts_length = 12
    num_batches = 64
    
    # 创建DGL图
    G = dgl.from_scipy(nx.to_scipy_sparse_array(nx.from_numpy_array(adj_matrix.numpy())))

    # 生成不同的时间序列数据，使用sin和cos
    t = torch.linspace(0, 2 * np.pi, ts_length)
    ts_sin = torch.stack([torch.sin(t + i) for i in range(num_nodes)], dim=0).transpose(0, 1).unsqueeze(0).repeat(num_batches, 1, 1)
    ts_cos = torch.stack([torch.cos(t + i) for i in range(num_nodes)], dim=0).transpose(0, 1).unsqueeze(0).repeat(num_batches, 1, 1)
    ts = (ts_sin + ts_cos) / 2  # 生成的时间序列数据
    
    # 划分子图并获取跨子图边及其时间序列特征
    subgraphs, split_indices, cross_edges = split_graph_3d(adj_matrix.numpy(), G, ts, num_subgrids)
    print("Subgraphs: ", len(subgraphs))  # 输出子图数量
    print("Split Indices: ", len(split_indices))
    print("Cross Edges: ", cross_edges.shape)
   
    # 输出每个子图的节点数
    for idx, indices in enumerate(split_indices):
        print(f"Subgraph {idx + 1} Node Count: {len(indices)}")
        
    # 对子图时间序列进行分割和合并
    ts_sublists = split_time_series(ts, split_indices)
    merged_ts = merge_time_series(ts_sublists, split_indices, num_nodes, num_batches, ts_length)

    # 检查合并的时间序列是否与原始时间序列相同
    if torch.equal(ts, merged_ts):
        print("时间序列分割和合并正确")
    else:
        print("时间序列分割和合并错误")

    # 可视化时间序列对比（第一个batch）
    visualize_time_series(
        ts[0, :, :], merged_ts[0, :, :],
        "Original Time Series (First Batch)", "Merged Time Series (First Batch)",
        save_path="view/time_series_comparison.png"
    )
    # 可视化子图
    # 可视化子图
    color_map, subgraph_indices = visualize_subgraphs_3d(subgraphs, cross_edges, num_subgrids, save_path="view/subgraph_3d.png")
    # 可视化原始图
    visualize_original_graph(adj_matrix.numpy(),  subgraph_indices, color_map, num_subgrids, save_path="view/original_graph.png")

    

    # 输出一些调试信息
   

    for i, subgraph in enumerate(subgraphs):
        num_edges = subgraph.number_of_edges()
        print(f"子图 {i+1} 边的数量: {num_edges}")
    
    # 输出跨子图的边的数量
    print(f"跨子图的边的数量: {cross_edges.shape[0]}")
