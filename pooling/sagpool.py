import torch
import torch.nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, AvgPooling, MaxPooling
from pooling.topkpool import TopKPooling, get_batch_id
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout


class SAGPool(torch.nn.Module):
    """The Self-Attention Pooling layer in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`

    Args:
        in_dim (int): The dimension of node feature.
        ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        conv_op (torch.nn.Module, optional): The graph convolution layer in dgl used to
        compute scale for each node. (default: :obj:`dgl.nn.GraphConv`)
        non_linearity (Callable, optional): The non-linearity function, a pytorch function.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_dim:int, ratio=0.5, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        if dgl.__version__ < "0.5":
            self.score_layer = GraphConv(in_dim, 1)
        else:
            self.score_layer = GraphConv(in_dim, 1, allow_zero_in_degree=True)
        self.non_linearity = non_linearity
        self.softmax = torch.nn.Softmax()

    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor, e_feat=None):
        score = self.score_layer(graph, feature).squeeze()
        perm, next_batch_num_nodes = TopKPooling(score, self.ratio, get_batch_id(graph.batch_num_nodes()), graph.batch_num_nodes())
        feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        graph = dgl.node_subgraph(graph, perm)

        # node_subgraph currently does not support batch-graph,
        # the 'batch_num_nodes' of the result subgraph is None.
        # So we manually set the 'batch_num_nodes' here.
        # Since global pooling has nothing to do with 'batch_num_edges',
        # we can leave it to be None or unchanged.
        graph.set_batch_num_nodes(next_batch_num_nodes)

        score = self.softmax(score)
        if torch.nonzero(torch.isnan(score)).size(0) > 0:
            print(score[torch.nonzero(torch.isnan(score))])
            raise KeyError

        if e_feat is not None:
            e_feat = graph.edata['feat'].unsqueeze(-1)

        return graph, feature, perm, score, e_feat


class SAGPoolBlock(torch.nn.Module):
    """A combination of GCN layer and SAGPool layer,
    followed by a concatenated (mean||sum) readout operation.
    return the graph embedding with the same dimension of node embedding.
    """
    def __init__(self, in_dim:int, pool_ratio=0.5):
        super(SAGPoolBlock, self).__init__()
        self.pool = SAGPool(in_dim, ratio=pool_ratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.mlp = torch.nn.Linear(in_dim * 2, in_dim)

    def forward(self, graph, feature):
        graph, out, _, _, _ = self.pool(graph, feature)
        g_out = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)
        g_out = F.relu(self.mlp(g_out))
        return graph, out, g_out


class SAGPoolReadout(torch.nn.Module):
    """A combination of GCN layer and SAGPool layer,
    followed by a concatenated (mean||sum) readout operation.
    """
    def __init__(self, net_params, pool_ratio=0.5, pool=True):
        super(SAGPoolReadout, self).__init__()
        in_dim = net_params['in_dim']
        out_dim = net_params['out_dim']
        dropout = net_params['dropout']
        n_classes = net_params['n_classes']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.e_feat = net_params['edge_feat']
        self.conv = GCNLayer(in_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual, e_feat=self.e_feat)
        self.use_pool = pool
        self.pool = SAGPool(out_dim, ratio=pool_ratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.MLP_layer = MLPReadout(out_dim * 2, n_classes)

    def forward(self, graph, feature, e_feat=None):
        out, e_feat = self.conv(graph, feature, e_feat)
        if self.use_pool:
            graph, out, _, _, e_feat = self.pool(graph, out, e_feat)
        hg = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)
        scores = self.MLP_layer(hg)
        return scores

    def loss(self, pred, label, cluster=False):

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss
