import torch
import torch.nn as nn
import torch.nn.functional as F


class GFS_node_classification(nn.Module):
    """
        PyTorch Implementation of GFS-node for Node Classification.
    """

    def __init__(self, nfeat, nclass, n_rules, n_depth, hidden_dim):
        super(GFS_node_classification, self).__init__()
        Depth_Layers = ([GFS_node_classification_Depth(nfeat, nclass, n_depth, hidden_dim) for _ in range(n_rules)])
        self.Depth_Layers = nn.ModuleList(Depth_Layers)

    def forward(self, x):
        res_list = []
        for Depth_Layer in self.Depth_Layers:
            origin_x = Depth_Layer(x)
            res_list.append(origin_x)
        return res_list


class GFS_node_classification_Depth(nn.Module):
    def __init__(self, nfeat, nclass, n_depth, hidden_dim, dropout=0.5):
        super(GFS_node_classification_Depth, self).__init__()
        self.dropout = dropout
        self.depth = n_depth
        self.sgc_list = [None] * self.depth
        self.sgc_list = nn.ModuleList(self.sgc_list)
        if self.depth == 1:
            self.sgc_list[self.depth - 1] = nn.Linear(nfeat, nclass)
            torch.nn.init.kaiming_normal(self.sgc_list[self.depth - 1].weight)
        else:
            for i in range(self.depth):
                if i == 0:
                    self.sgc_list[i] = nn.Linear(nfeat, hidden_dim)
                    torch.nn.init.kaiming_normal(self.sgc_list[i].weight)
                elif i == self.depth - 1:
                    self.sgc_list[i] = nn.Linear(hidden_dim, nclass)
                    torch.nn.init.kaiming_normal(self.sgc_list[i].weight)
                else:
                    self.sgc_list[i] = nn.Linear(hidden_dim, hidden_dim)
                    torch.nn.init.kaiming_normal(self.sgc_list[i].weight)

    def forward(self, x):
        if self.depth == 1:
            y = self.sgc_list[self.depth - 1](x)
            y_pred = F.log_softmax(y, dim=1)
            return y_pred
        else:
            for i in range(self.depth - 1):
                x = self.sgc_list[i](x)
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
            y = self.sgc_list[self.depth - 1](x)
            y_pred = F.log_softmax(y, dim=1)
            return y_pred


class GFS_node_regression(nn.Module):
    """
        PyTorch Implementation of GFS-node for Node Regression.
    """

    def __init__(self, nfeat, n_rules, n_depth, hidden_dim):
        super(GFS_node_regression, self).__init__()
        Depth_Layers = ([GFS_node_regression_Depth(nfeat, n_depth, hidden_dim) for _ in range(n_rules)])
        self.Depth_Layers = nn.ModuleList(Depth_Layers)

    def forward(self, x):
        res_list = []
        for Depth_Layer in self.Depth_Layers:
            origin_x = Depth_Layer(x)
            res_list.append(origin_x)
        return res_list


class GFS_node_regression_Depth(nn.Module):
    def __init__(self, nfeat, n_depth, hidden_dim, dropout=0.5):
        super(GFS_node_regression_Depth, self).__init__()
        self.dropout = dropout
        self.depth = n_depth
        self.LMPN_list = [None] * self.depth
        self.LMPN_list = nn.ModuleList(self.LMPN_list)
        if self.depth == 1:
            self.LMPN_list[self.depth - 1] = nn.Linear(nfeat, 1)
            torch.nn.init.kaiming_normal(self.LMPN_list[self.depth - 1].weight)
        else:
            for i in range(self.depth):
                if i == 0:
                    self.LMPN_list[i] = nn.Linear(nfeat, hidden_dim)
                    torch.nn.init.kaiming_normal(self.LMPN_list[i].weight)
                elif i == self.depth - 1:
                    self.LMPN_list[i] = nn.Linear(hidden_dim, 1)
                    torch.nn.init.kaiming_normal(self.LMPN_list[i].weight)
                else:
                    self.LMPN_list[i] = nn.Linear(hidden_dim, hidden_dim)
                    torch.nn.init.kaiming_normal(self.LMPN_list[i].weight)

    def forward(self, x):
        if self.depth == 1:
            y = self.LMPN_list[self.depth - 1](x)
            y_pred = y
            return y_pred
        else:
            for i in range(self.depth - 1):
                x = self.LMPN_list[i](x)
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
            y = self.LMPN_list[self.depth - 1](x)
            y_pred = y
            return y_pred