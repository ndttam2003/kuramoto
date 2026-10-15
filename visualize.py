import torch
import torch_geometric
import os
from train_qm9 import get_args
from source.modules.graph_task import KuramotoGNN
from ema_pytorch import EMA
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F

args = get_args()

from pprint import pprint

pprint(vars(args))

model = KuramotoGNN(
        in_dim=11, 
        osc_dim=args.osc_dim, 
        final_dim=1, 
        L=args.L, 
        T=args.T, 
        hidden_dim=args.hidden_dim,
        message_passing_type = args.message_passing_type, 
        num_head=args.num_head, 
        gamma=args.gamma, 
        edge_dim=0, 
        readout_type=args.readout_type
    )

checkpoint = torch.load(
    "model.zip",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(checkpoint)
model.eval()

ema = EMA(model, beta=args.beta_ema, update_every=10, update_after_step=200)

checkpoint = torch.load(
    "ema_model.zip",
    map_location="cpu",
    weights_only=True,
)
ema.load_state_dict(checkpoint)

qm9_dataset = torch_geometric.datasets.QM9(root=args.data_path)
print(qm9_dataset)


def visualize_pyg_spring_layout(
    x,
    edge_index,
    raw_labels,
    sim_threshold: float = 0.2,
    node_size: int = 150,
    seed: int = 42,
    show_edge_weights: bool = True,
    show_node_labels: bool = True,
    
):
    """Trực quan hóa sự đồng bộ pha trên PyG Data bằng NetworkX Spring Layout.

    Args:
        data (Data): Đối tượng PyG chứa `x` [num_nodes, D] và `edge_index` [2,
          num_edges]
        sim_threshold (float): Ngưỡng tương đồng cosine. Cạnh có similarity <
          threshold sẽ bị ngắt bỏ để các cụm tách nhau ra.
        node_size (int): Kích thước hiển thị của node.
        seed (int): Seed cố định layout để kết quả nhất quán.
    """
    # 1. Chuyển tensor sang CPU numpy
    # x = data.x.detach().cpu()
    # edge_index = data.edge_index.detach().cpu()
    # x = x.detach().numpy()
    # edge_index = edge_index.detach().numpy()
    num_nodes, D = x.shape

    # 2. Chuẩn hóa L2 để vector pha nằm trên mặt cầu đơn vị
    # print(torch.linalg.norm(x.reshape(num_nodes,-1, args.osc_dim), dim=-1))
    x_norm = x #F.normalize(x.reshape(num_nodes,-1, args.osc_dim), p=2, dim=-1)

    # 3. Khởi tạo đồ thị NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # 4. Tính toán trọng số cạnh (Cosine Similarity)
    # Lấy danh sách các cặp đỉnh (u, v)
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()

    edge_weights = []
    edges_to_draw = []

    x_osc = x.reshape(num_nodes,-1, args.osc_dim)
    for u, v in zip(src, dst):
        if u < v:  # Đồ thị vô hướng: chỉ xét mỗi cặp một lần
            # Tích vô hướng của 2 vector đã chuẩn hóa L2 chính là Cosine Similarity
            # print(torch.sum(x_osc[u] * x_osc[v], dim=-1).mean())
            sim = float(torch.sum(x_osc[u] * x_osc[v], dim=-1).mean().item())
            print(sim)

            # Chỉ giữ cạnh nếu 2 node có độ đồng bộ cao hơn ngưỡng
            if sim > sim_threshold:
                sim = sim * 5
                G.add_edge(u, v, weight=sim)
                edges_to_draw.append((u, v))
                edge_weights.append(sim)

    # 5. Mã hóa màu sắc các Node qua PCA (D-dim -> 3-dim RGB)
    if D >= 3:
        pca = PCA(n_components=3)
        colors = pca.fit_transform(x_norm.detach().numpy())
        # Chuẩn hóa min-max về khoảng [0, 1] cho màu RGB
        colors = (colors - colors.min(0)) / (
            colors.max(0) - colors.min(0) + 1e-8
        )
    elif D == 2:
        # Nếu D=2: Dùng góc pha trực tiếp để lấy màu trên colormap hsv
        angles = torch.atan2(x_norm[:, 1], x_norm[:, 0]).detach().numpy()
        norm_angles = (angles + np.pi) / (2 * np.pi)
        colors = plt.cm.hsv(norm_angles)[:, :3]
    else:
        # D=1
        colors = plt.cm.viridis(x_norm[:, 0].detach().numpy())[:, :3]

    # 6. Tính toán vị trí Spring Layout dựa trên độ đồng bộ (weight)
    # Các cạnh có weight lớn (đồng bộ cao) sẽ kéo các node dính chặt lại với nhau
    pos = nx.spring_layout(G, weight="weight", seed=seed, iterations=100)

    # 7. Vẽ đồ thị
    fig, ax = plt.subplots(figsize=(9, 8))

    # Vẽ Nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=colors, node_size=node_size, alpha=0.9, ax=ax
    )

    labels_dict = {i: str(raw_labels[i]) for i in range(num_nodes)}
    if show_node_labels:
        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels_dict,
            font_size=9,
            font_color="black",
            font_weight="bold",
            ax=ax,
        )

    # Vẽ Edges (độ đậm và độ mờ tỉ lệ thuận với độ đồng bộ)
    if edges_to_draw:
        widths = [w * 2.5 for w in edge_weights]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges_to_draw,
            width=widths,
            edge_color=edge_weights,
            edge_cmap=plt.cm.Greys,
            edge_vmin=0.0,
            edge_vmax=1.0,
            alpha=0.6,
            ax=ax,
        )

    edge_labels = {
        (u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges_to_draw
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color="darkblue",
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc="white",
            ec="none",
            alpha=0.7,  # Nền trắng mờ để số không bị đè lên nét vẽ cạnh
        ),
        ax=ax,
    )

    ax.set_title(
        f"Spring Layout Phase Synchronization\n(Edges with Cosine Sim > {sim_threshold})",
        fontsize=13,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()

sample = qm9_dataset[100]
print(sample)
from rdkit.Chem import Draw
from rdkit import Chem
m = Chem.MolFromSmiles(sample.smiles)
img = Draw.MolToImage(m)
plt.imshow(img)

print(sample.x)
outputs, track_osc = model(x = sample.x, edge_index = sample.edge_index, batch = None)

print(outputs.shape)
print([type(i) for i in track_osc])
for i in track_osc[1:]:
    visualize_pyg_spring_layout(
        x = i[-1],
        edge_index=sample.edge_index,
        raw_labels=sample.x[:, 5]
    )