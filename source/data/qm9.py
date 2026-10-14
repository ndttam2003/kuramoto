import torch
import torch_geometric
import numpy as np
from torch_geometric.loader import DataLoader


def cal_relative_matrix(coord, edge_index):
    # neighbor_coord = []
    # print(vmap_coor(edge_index[0], edge_index[1]))
    # for src, target in edge_index.T:
    #     dist = (coord[target] - coord[src])
    #     print(dist.shape)
    #     neighbor_coord.append(dist)
    # print(neighbor_coord)
    # return torch.vstack(neighbor_coord)
    vmap_coor = torch.func.vmap(lambda s, t : coord[t] - coord[s])
    return vmap_coor(edge_index[0], edge_index[1])

def transfrom_data(data):
    z = data.z
    # data.feature = z_to_15_features(z)
    relatix_matrix = cal_relative_matrix(data.pos, data.edge_index)
    data.relatix_matrix = relatix_matrix
    return data

def get_train_val_test(BATCH_SIZE, num_workers):
    data_path = "/home/tam/code_base/my_egnn/"

    qm9_dataset = torch_geometric.datasets.QM9(root=data_path)
    qm9_dataset_transformed = qm9_dataset
    # qm9_dataset_transformed = [transfrom_data(x) for x in qm9_dataset ]
    # np.random.seed(0)
    # print("Warning seed is constant")
    perm = torch.from_numpy(np.random.permutation(len(qm9_dataset_transformed)))

    train_idx = perm[:100000]
    index_test = int(0.1 * len(qm9_dataset_transformed)) + 1
    test_idx   = perm[100000:100000 + index_test]
    val_idx  = perm[100000 + index_test:]

    train_dataset = qm9_dataset_transformed[train_idx]
    val_dataset   = qm9_dataset_transformed[val_idx]
    test_dataset  = qm9_dataset_transformed[test_idx]
    # train_dataset = [qm9_dataset_transformed[i] for i in train_idx]
    # val_dataset   = [qm9_dataset_transformed[i] for i in val_idx]
    # test_dataset  = [qm9_dataset_transformed[i] for i in test_idx]
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    t, v, ts = get_train_val_test(8)
    print(next(iter(t)))
    print(next(iter(v)))
    print(next(iter(ts)))

