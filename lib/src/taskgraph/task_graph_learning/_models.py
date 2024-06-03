import torch
import torch.nn as nn
import numpy as np

from ._loss import task_graph_maximum_likelihood_loss


class TaskGraph(nn.Module):
    """Task Graph model"""
    def __init__(self, num_nodes):
        super(TaskGraph, self).__init__()
        self.task_graph = nn.Parameter(torch.rand(num_nodes, num_nodes))

    def forward(self):
        return self.task_graph
    

class DO(nn.Module):
    """Direct Optimization model"""
    def __init__(self, num_nodes, device='cpu'):
        super(DO, self).__init__()
        self.adjacency_matrix = TaskGraph(num_nodes)
        self.softmax = nn.Softmax(dim=1)
        self.size = num_nodes
        self.device = device

    def get_adjacency_matrix(self):
        mask = np.ones((self.size, self.size), dtype=bool)
        mask[0, :] = False
        mask[1, :] = False
        mask[1, 0] = True
        mask[:, -1] = False
        mask[:, -2] = False
        mask[-1, :] = False
        mask[-1, -2] = True
        action_mask = (mask) & (~np.identity(self.size, dtype=bool))
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        values = self.adjacency_matrix()
        softmax_input = torch.where(action_mask, values, torch.tensor(-float('inf')).to(self.device))
        values = self.softmax(softmax_input)
        return values
        
    def forward(self, y, eps=1e-6, beta=0.005):
        values = self.get_adjacency_matrix()
        # We sum eps to avoid log(0)
        return task_graph_maximum_likelihood_loss(y, values + eps, beta)