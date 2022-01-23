import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import dgl
from dgl.nn import GINConv
from scipy import sparse
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, enhance=False):
        super(MLP, self).__init__()

        self.enhance = enhance

        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        if enhance:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        if self.enhance:
            x = self.bn1(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)

        x = self.fc2(x)
        if self.enhance:
            x = self.bn2(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)

        x = self.fc3(x)

        return x


class MEWISPool(nn.Module):
    def __init__(self, hidden_dim):
        super(MEWISPool, self).__init__()

        self.gc1 = GINConv(MLP(1, hidden_dim, hidden_dim), 'sum')
        self.gc2 = GINConv(MLP(hidden_dim, hidden_dim, hidden_dim), 'sum')
        self.gc3 = GINConv(MLP(hidden_dim, hidden_dim, 1), 'sum')

    def forward(self, g, x, batch):
        # computing the graph laplacian and adjacency matrix
        batch_nodes = x.size(0)
        if g.num_edges() != 0:
            # L_indices, L_values = get_laplacian(edge_index)
            # L = torch.sparse.FloatTensor(L_indices, L_values, torch.Size([batch_nodes, batch_nodes]))
            # A = torch.diag(torch.diag(L.to_dense())) - L.to_dense()

            num_nodes = g.number_of_nodes()
            A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
            L = sparse.eye(num_nodes) - A

            A = self.matrix2tensor(A, device=x.device)
            L = self.matrix2tensor(L, device=x.device)

            # entropy computation
            entropies = self.compute_entropy(x, L, A, batch)  # Eq. (8)
        else:
            A = torch.zeros([batch_nodes, batch_nodes])
            norm = torch.norm(x, dim=1).unsqueeze(-1)
            entropies = norm / norm

        # graph convolution and probability scores
        probabilities = self.gc1(g, entropies)
        probabilities = self.gc2(g, probabilities)
        probabilities = self.gc3(g, probabilities)
        probabilities = torch.sigmoid(probabilities)

        # conditional expectation; Algorithm 1
        gamma = entropies.sum()
        loss = self.loss_fn(entropies, probabilities, A, gamma)  # Eq. (9)

        mewis = self.conditional_expectation(entropies, probabilities, A, loss, gamma)

        # graph reconstruction; Eq. (10)
        g, h, adj_pooled = self.graph_reconstruction(mewis, x, A, device=x.device)
        edge_index_pooled, batch_pooled = self.to_edge_index(adj_pooled, mewis, batch)

        return g, h, loss, batch_pooled

    @staticmethod
    def matrix2tensor(csr_matrix, device):
        coo_matrix = csr_matrix.tocoo()
        values = coo_matrix.data
        indices = np.vstack((coo_matrix.row, coo_matrix.col))

        i = torch.tensor(indices, device=device).long()
        v = torch.tensor(values, device=device)
        shape = coo_matrix.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().to(torch.float32)

    @staticmethod
    def compute_entropy(x, L, A, batch):
        # computing local variations; Eq. (5)
        V = x * torch.matmul(L, x) - x * torch.matmul(A, x) + torch.matmul(A, x * x)
        V = torch.norm(V, dim=1)

        # computing the probability distributions based on the local variations; Eq. (7)
        P = torch.cat([torch.softmax(V[batch == i], dim=0) for i in torch.unique(batch)])
        P[P == 0.] += 1
        # computing the entropies; Eq. (8)
        H = -P * torch.log(P)

        return H.unsqueeze(-1)

    @staticmethod
    def loss_fn(entropies, probabilities, A, gamma):
        term1 = -torch.matmul(entropies.t(), probabilities)[0, 0]

        term2 = torch.matmul(torch.matmul(probabilities.t(), A), probabilities).sum()

        return gamma + term1 + term2

    def conditional_expectation(self, entropies, probabilities, A, threshold, gamma):
        sorted_probabilities = torch.sort(probabilities, descending=True, dim=0)

        dummy_probabilities = probabilities.detach().clone()
        selected = set()
        rejected = set()

        for i in range(sorted_probabilities.values.size(0)):
            node_index = sorted_probabilities.indices[i].item()
            neighbors = torch.where(A[node_index] == 1)[0]
            if len(neighbors) == 0:
                selected.add(node_index)
                continue
            if node_index not in rejected and node_index not in selected:
                s = dummy_probabilities.clone()
                s[node_index] = 1
                s[neighbors] = 0

                loss = self.loss_fn(entropies, s, A, gamma)

                if loss <= threshold:
                    selected.add(node_index)
                    for n in neighbors.tolist():
                        rejected.add(n)

                    dummy_probabilities[node_index] = 1
                    dummy_probabilities[neighbors] = 0

        mewis = list(selected)
        mewis = sorted(mewis)

        return mewis

    def graph_reconstruction(self, mewis, x, A, device):
        x_pooled = x[mewis]

        A2 = torch.matmul(A, A)
        A3 = torch.matmul(A2, A)

        A2 = A2[mewis][:, mewis]
        A3 = A3[mewis][:, mewis]

        I = torch.eye(len(mewis), device=device)
        one = torch.ones([len(mewis), len(mewis)], device=device)

        adj_pooled = (one - I) * torch.clamp(A2 + A3, min=0, max=1)

        src, dst = np.nonzero(adj_pooled.cpu().numpy())
        g = dgl.graph((src, dst), num_nodes=adj_pooled.size(0)).to(device)
        g.ndata['h'] = x_pooled

        return g, x_pooled, adj_pooled

    @staticmethod
    def to_edge_index(adj_pooled, mewis, batch):
        row1, row2 = torch.where(adj_pooled > 0)
        edge_index_pooled = torch.cat([row1.unsqueeze(0), row2.unsqueeze(0)], dim=0)
        batch_pooled = batch[mewis]

        return edge_index_pooled, batch_pooled


class MEWISPoolNet(nn.Module):
    def __init__(self, net_params, readout='sum'):
        super(MEWISPoolNet, self).__init__()

        input_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        self.e_feat = net_params['edge_feat']

        self.gc1 = GINConv(MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True), 'sum')
        self.pool1 = MEWISPool(hidden_dim=hidden_dim)
        self.gc2 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True), 'sum')
        self.pool2 = MEWISPool(hidden_dim=hidden_dim)
        self.gc3 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True), 'sum')
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=n_classes)

        if readout == 'sum':
            self.pool = SumPooling()
        elif readout == 'mean':
            self.pool = AvgPooling()
        elif readout == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h, e):
        batch = []
        for i, g_n_nodes in enumerate(g.batch_num_nodes()):
            batch += [i] * g_n_nodes
        batch = torch.tensor(batch, device=h.device)

        x = torch.relu(self.gc1(g, h))
        g_pooled1, x_pooled1, loss1, batch_pooled1 = self.pool1(g, x, batch)
        x_pooled1 = torch.relu(self.gc2(g_pooled1, x_pooled1))
        g_pooled2, x_pooled2, loss2, batch_pooled2 = self.pool2(g_pooled1, x_pooled1, batch_pooled1)
        x_pooled2 = self.gc3(g_pooled2, x_pooled2)

        # readout = torch.cat([self.pool(g, x), self.pool(g_pooled1, x_pooled1), self.pool(g_pooled2, x_pooled2)], dim=1)
        # readout = self.pool(g_pooled2, x_pooled2)
        readout = torch.cat([x_pooled2[batch_pooled2 == i].mean(0).unsqueeze(0) for i in torch.unique(batch_pooled2)], dim=0)

        out = torch.relu(self.fc1(readout))
        out = self.fc2(out)

        return torch.log_softmax(out, dim=-1), loss1 + loss2

    def loss(self, pred, label, loss_pool):
        criterion = nn.CrossEntropyLoss()
        loss_classification = criterion(pred, label.view(-1))
        loss = loss_classification + 0.01 * loss_pool
        return loss


class Net2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Net2, self).__init__()

        self.gc1 = GINConv(MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.gc2 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.gc3 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.pool1 = MEWISPool(hidden_dim=hidden_dim)
        self.gc4 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, x, edge_index, batch):
        x = self.gc1(x, edge_index)
        x = torch.relu(x)

        x = self.gc2(x, edge_index)
        x = torch.relu(x)

        x = self.gc3(x, edge_index)
        x = torch.relu(x)
        readout2 = torch.cat([x[batch == i].mean(0).unsqueeze(0) for i in torch.unique(batch)], dim=0)

        x_pooled1, edge_index_pooled1, batch_pooled1, loss1, mewis = self.pool1(x, edge_index, batch)

        x_pooled1 = self.gc4(x_pooled1, edge_index_pooled1)

        readout = torch.cat([x_pooled1[batch_pooled1 == i].mean(0).unsqueeze(0) for i in torch.unique(batch_pooled1)],
                            dim=0)

        out = readout2 + readout

        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)

        return torch.log_softmax(out, dim=-1), loss1


class Net3(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Net3, self).__init__()

        self.gc1 = GINConv(MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.gc2 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.pool1 = MEWISPool(hidden_dim=hidden_dim)
        self.gc3 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.gc4 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.pool2 = MEWISPool(hidden_dim=hidden_dim)
        self.gc5 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, x, edge_index, batch):
        x = self.gc1(x, edge_index)
        x = torch.relu(x)

        x = self.gc2(x, edge_index)
        x = torch.relu(x)

        x_pooled1, edge_index_pooled1, batch_pooled1, loss1, mewis1 = self.pool1(x, edge_index, batch)

        x_pooled1 = self.gc3(x_pooled1, edge_index_pooled1)
        x_pooled1 = torch.relu(x_pooled1)

        x_pooled1 = self.gc4(x_pooled1, edge_index_pooled1)
        x_pooled1 = torch.relu(x_pooled1)

        x_pooled2, edge_index_pooled2, batch_pooled2, loss2, mewis2 = self.pool2(x_pooled1, edge_index_pooled1,
                                                                                 batch_pooled1)

        x_pooled2 = self.gc5(x_pooled2, edge_index_pooled2)
        x_pooled2 = torch.relu(x_pooled2)

        readout = torch.cat([x_pooled2[batch_pooled2 == i].mean(0).unsqueeze(0) for i in torch.unique(batch_pooled2)],
                            dim=0)

        out = self.fc1(readout)
        out = torch.relu(out)
        out = self.fc2(out)

        return torch.log_softmax(out, dim=-1), loss1 + loss2